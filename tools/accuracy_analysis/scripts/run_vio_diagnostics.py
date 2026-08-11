#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
DIAGNOSTIC_FILES = (
    "vio_diag_metadata.csv",
    "vio_diag_frame.csv",
    "vio_diag_triangulation.csv",
    "vio_diag_initialisation.csv",
    "vio_diag_ransac.csv",
    "vio_diag_landmark_events.csv",
)
SEQUENCE_PATTERN = re.compile(r"^\d{8}-\d{6}$")


@dataclass(frozen=True)
class ReplayInputs:
    experiment_id: str
    sequence: str
    intervention: str
    intervention_value: str
    run_name: str
    dataset: Path
    mocap: Path
    run_dir: Path
    diagnostics_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class DatasetExperiment:
    experiment_id: str
    dataset: Path
    sequence: str
    intervention: str
    intervention_value: str


def parse_image_delay(path: Path) -> float:
    pattern = re.compile(r"^\s*image_delay:\s*([0-9.eE+-]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    raise ValueError(f"{path}: image_delay not found")


def repository_build_id(repository: Path = REPOSITORY) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{head}-dirty" if dirty.strip() else head


def resolve_mocap(
    reference_results_root: Path, sequence: str
) -> Path:
    day = sequence.split("-", 1)[0]
    candidates = sorted(
        path.resolve()
        for path in (reference_results_root / day).rglob("mocap*.log")
        if sequence in path.parts
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{sequence}: expected exactly one mocap*.log inside its "
            f"sequence directory, found {len(candidates)}"
        )
    return candidates[0]


def discover_reference_sequences(reference_results_root: Path) -> list[str]:
    if not reference_results_root.is_dir():
        raise ValueError(
            f"missing reference results root: {reference_results_root}"
        )
    sequences = sorted(
        {
            path.parent.name
            for path in reference_results_root.rglob("mocap*.log")
            if SEQUENCE_PATTERN.fullmatch(path.parent.name)
        }
    )
    if not sequences:
        raise ValueError(
            f"no reference sequences with mocap*.log found under "
            f"{reference_results_root}"
        )
    return sequences


def resolve_dataset(data_root: Path, sequence: str) -> Path:
    day = sequence.split("-", 1)[0]
    candidates = sorted(
        path.resolve()
        for path in (data_root / day).rglob(f"{sequence}_euroc")
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{sequence}: expected exactly one dataset directory named "
            f"{sequence}_euroc under {data_root / day}, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def validate_output_target(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"refusing nonempty output directory: {run_dir}")


def read_dataset_manifest(path: Path) -> list[DatasetExperiment]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing dataset manifest: {path}")
    required = (
        "experiment_id",
        "dataset",
        "sequence",
        "intervention",
        "intervention_value",
    )
    experiments: list[DatasetExperiment] = []
    experiment_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != required:
            raise ValueError(
                f"{path}: expected manifest fields {list(required)}, "
                f"found {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            values = {name: (row.get(name) or "").strip() for name in required}
            for name, value in values.items():
                if not value:
                    raise ValueError(f"{path}:{line_number}: empty {name}")
            experiment_id = values["experiment_id"]
            if experiment_id in experiment_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate experiment_id "
                    f"{experiment_id}"
                )
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
                raise ValueError(
                    f"{path}:{line_number}: unsafe experiment_id "
                    f"{experiment_id}"
                )
            experiment_ids.add(experiment_id)
            dataset = Path(values["dataset"])
            if not dataset.is_absolute():
                dataset = path.parent / dataset
            experiments.append(
                DatasetExperiment(
                    experiment_id=experiment_id,
                    dataset=dataset.resolve(),
                    sequence=values["sequence"],
                    intervention=values["intervention"],
                    intervention_value=values["intervention_value"],
                )
            )
    if not experiments:
        raise ValueError(f"{path}: dataset manifest is empty")
    return experiments


def replay_experiments(arguments: argparse.Namespace) -> list[DatasetExperiment]:
    sequences = getattr(arguments, "sequences", None)
    dataset_manifest = getattr(arguments, "dataset_manifest", None)
    resume_all = getattr(arguments, "resume_all", False)
    if sum((bool(sequences), bool(dataset_manifest), bool(resume_all))) != 1:
        raise ValueError(
            "exactly one of --sequences, --dataset-manifest, and "
            "--resume-all is required"
        )
    if dataset_manifest:
        return read_dataset_manifest(dataset_manifest)
    if resume_all:
        sequences = discover_reference_sequences(
            arguments.reference_results_root
        )
    return [
        DatasetExperiment(
            experiment_id=sequence,
            dataset=(
                resolve_dataset(arguments.data_root, sequence)
                if resume_all
                else (
                    arguments.data_root
                    / sequence.split("-", 1)[0]
                    / f"{sequence}_euroc"
                ).resolve()
            ),
            sequence=sequence,
            intervention="baseline",
            intervention_value="none",
        )
        for sequence in sequences
    ]


def prepare_replays(arguments: argparse.Namespace) -> list[ReplayInputs]:
    binary = arguments.binary.resolve()
    config = arguments.config.resolve()
    if not binary.is_file():
        raise ValueError(f"missing binary: {binary}")
    if not config.is_file():
        raise ValueError(f"missing config: {config}")
    parse_image_delay(config)

    manifest_mode = bool(getattr(arguments, "dataset_manifest", None))
    resume_all = bool(getattr(arguments, "resume_all", False))
    skip_complete = bool(
        getattr(arguments, "skip_complete", False) or resume_all
    )
    prepared: list[ReplayInputs] = []
    completed_runs = 0
    experiments = replay_experiments(arguments)
    for experiment in experiments:
        sequence = experiment.sequence
        dataset = experiment.dataset
        if not dataset.is_dir():
            raise ValueError(f"missing dataset: {dataset}")
        mocap = resolve_mocap(arguments.reference_results_root, sequence)
        for repeat in range(1, arguments.repeats + 1):
            run_name = f"run{repeat}"
            output_id = experiment.experiment_id if manifest_mode else sequence
            run_dir = (arguments.results_root / output_id / run_name).resolve()
            diagnostics_dir = run_dir / "diagnostics"
            replay = ReplayInputs(
                experiment_id=experiment.experiment_id,
                sequence=sequence,
                intervention=experiment.intervention,
                intervention_value=experiment.intervention_value,
                run_name=run_name,
                dataset=dataset,
                mocap=mocap,
                run_dir=run_dir,
                diagnostics_dir=diagnostics_dir,
                command=(
                    str(binary),
                    str(config),
                    str(dataset),
                    str(run_dir),
                ),
            )
            if run_dir.exists() and any(run_dir.iterdir()):
                if not skip_complete:
                    validate_output_target(run_dir)
                try:
                    validate_completed_run(replay)
                except ValueError as error:
                    raise ValueError(
                        f"{run_dir}: not a valid completed run: {error}"
                    ) from error
                completed_runs += 1
                print(f"skipping completed {experiment.experiment_id}/{run_name}")
                continue
            prepared.append(replay)
    if resume_all:
        print(
            f"resume summary: sequences={len(experiments)}, "
            f"completed_runs={completed_runs}, "
            f"pending_runs={len(prepared)}"
        )
    return prepared


def diagnostic_metadata(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"schema_version", "key", "value"}:
            raise ValueError(f"invalid metadata header: {path}")
        return {row["key"]: row["value"] for row in reader}


def validate_completed_run(replay: ReplayInputs) -> list[str]:
    diagnostics = replay.diagnostics_dir
    missing = [
        name for name in DIAGNOSTIC_FILES if not (diagnostics / name).is_file()
    ]
    if missing:
        raise ValueError(f"{diagnostics}: missing diagnostic files {missing}")
    if not (diagnostics / ".vio_diagnostics.complete").is_file():
        raise ValueError(f"{diagnostics}: missing completion sentinel")
    if (diagnostics / ".vio_diagnostics.active").exists():
        raise ValueError(f"{diagnostics}: active sentinel remains")
    metadata = diagnostic_metadata(diagnostics / "vio_diag_metadata.csv")
    if metadata.get("run_complete") != "true":
        raise ValueError(f"{diagnostics}: run_complete=true is absent")
    if metadata.get("writer_failed", "false") == "true":
        raise ValueError(f"{diagnostics}: diagnostics writer reported failure")

    online = sorted(
        path
        for path in replay.run_dir.glob("okvis2-*_trajectory.csv")
        if "-final-ba_trajectory.csv" not in path.name
    )
    final_ba = sorted(replay.run_dir.glob("okvis2-*-final-ba_trajectory.csv"))
    if not online:
        raise ValueError(f"{replay.run_dir}: online trajectory is absent")
    if not final_ba:
        raise ValueError(f"{replay.run_dir}: final-BA trajectory is absent")
    return sorted(
        str(path.relative_to(replay.run_dir))
        for path in replay.run_dir.rglob("*")
        if path.is_file()
    )


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_replay(
    replay: ReplayInputs,
    config: Path,
    build_id: str,
) -> None:
    replay.run_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OKVIS_DIAGNOSTICS_DIR": str(replay.diagnostics_dir),
            "OKVIS_DIAGNOSTICS_RUN_ID":
                f"{replay.experiment_id}-{replay.run_name}",
            "OKVIS_DIAGNOSTICS_BUILD_ID": build_id,
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    start = dt.datetime.now(dt.timezone.utc)
    log_path = replay.run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(replay.command),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    end = dt.datetime.now(dt.timezone.utc)
    if result.returncode != 0:
        raise RuntimeError(
            f"{replay.sequence}/{replay.run_name}: process exited "
            f"with {result.returncode}; see {log_path}"
        )
    produced_files = validate_completed_run(replay)
    manifest = {
        "schema_version": 1,
        "experiment_id": replay.experiment_id,
        "dataset": str(replay.dataset),
        "sequence": replay.sequence,
        "intervention": replay.intervention,
        "intervention_value": replay.intervention_value,
        "run": replay.run_name,
        "command": list(replay.command),
        "dataset_path": str(replay.dataset),
        "mocap_path": str(replay.mocap),
        "config_path": str(config.resolve()),
        "image_delay_s": parse_image_delay(config),
        "build_id": build_id,
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "return_code": result.returncode,
        "produced_files": produced_files,
    }
    manifest_path = replay.run_dir / "run_manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"completed {replay.experiment_id}/{replay.run_name}: {manifest_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay OKVIS with structured causal diagnostics"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("/home/chenguyuan/code/okvis_ws/build/okvis/")
        / "okvis_app_synchronous",
    )
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY / "config/okvis2_eucm_EGO2.yaml"
    )
    parser.add_argument("--data-root", type=Path, default=Path("/home/chenguyuan/data"))
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        default=REPOSITORY / "workspace/ego2_results",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPOSITORY / "workspace/ego2_results/202608_causal_diagnostics",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequences", nargs="+")
    source.add_argument("--dataset-manifest", type=Path)
    source.add_argument(
        "--resume-all",
        action="store_true",
        help=(
            "discover all reference sequences, skip valid completed runs, "
            "and schedule only missing runs"
        ),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="skip valid completed run directories; reject partial outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.repeats < 1:
        raise ValueError("--repeats must be positive")
    if arguments.jobs != 1:
        raise ValueError("only --jobs 1 is currently supported")
    replays = prepare_replays(arguments)
    if arguments.dry_run:
        for replay in replays:
            print(" ".join(replay.command))
        return 0
    build_id = repository_build_id()
    for replay in replays:
        run_replay(replay, arguments.config, build_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
