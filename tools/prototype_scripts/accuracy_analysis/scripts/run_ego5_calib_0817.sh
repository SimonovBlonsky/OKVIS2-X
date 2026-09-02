#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

BINARY=${BINARY:-${REPOSITORY}/../../build/okvis/okvis_app_synchronous}
CONFIG=${CONFIG:-${REPOSITORY}/config/okvis2_eucm_EGO5_new.yaml}
DATA_ROOTS=(
  "${DATA_ROOT_1:-/home/chenguyuan/data/20260812}"
  "${DATA_ROOT_2:-/home/chenguyuan/data/20260813_ego5}"
)
RESULTS_ROOT=${RESULTS_ROOT:-${REPOSITORY}/workspace/ego5_results/calib_0817}
REPEATS=${REPEATS:-2}
DRY_RUN=${DRY_RUN:-false}

usage() {
  cat <<'EOF'
Run every EuRoC sequence in the 20260812 and 20260813_ego5 data roots
twice with config/okvis2_eucm_EGO5_new.yaml.

Usage:
  run_ego5_calib_0817.sh [--dry-run]

Options:
  --dry-run   Print pending OKVIS commands without running them
  -h, --help  Show this help

Results are written to
workspace/ego5_results/calib_0817/<sequence>/run{1,2}.
Existing nonempty run directories are skipped to avoid overwriting results.
VIO diagnostics are not enabled.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

is_complete_euroc() {
  local dataset=$1
  local camera

  [[ -f ${dataset}/.complete && -f ${dataset}/imu0/data.csv ]] || return 1
  for camera in 0 1 2 3; do
    [[ -f ${dataset}/cam${camera}/data.csv ]] || return 1
    [[ -d ${dataset}/cam${camera}/data ]] || return 1
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -x ${BINARY} ]] || die "missing executable: ${BINARY}"
[[ -f ${CONFIG} ]] || die "missing config: ${CONFIG}"

declare -a datasets=()
declare -A sequence_sources=()
for data_root in "${DATA_ROOTS[@]}"; do
  [[ -d ${data_root} ]] || die "missing data root: ${data_root}"

  mapfile -d '' -t root_datasets < <(
    find "${data_root}" -mindepth 1 -maxdepth 1 -type d -name '*_euroc' \
      -print0 | sort -z
  )
  ((${#root_datasets[@]} > 0)) || \
    die "no *_euroc datasets found in ${data_root}"

  for dataset in "${root_datasets[@]}"; do
    dataset_name=${dataset##*/}
    sequence=${dataset_name%_euroc}
    is_complete_euroc "${dataset}" || \
      die "incomplete or invalid EuRoC dataset: ${dataset}"
    if [[ -n ${sequence_sources[${sequence}]+x} ]]; then
      die "duplicate sequence ${sequence}: ${sequence_sources[${sequence}]} and ${dataset}"
    fi
    sequence_sources[${sequence}]=${dataset}
    datasets+=("${dataset}")
  done
done

printf 'Found %d complete sequence(s) across %d data roots; each sequence will run %d times.\n' \
  "${#datasets[@]}" "${#DATA_ROOTS[@]}" "${REPEATS}"
printf 'Config:  %s\nResults: %s\n' "${CONFIG}" "${RESULTS_ROOT}"

completed=0
skipped=0
failed=0
planned=0
declare -a failed_runs=()

for dataset in "${datasets[@]}"; do
  dataset_name=${dataset##*/}
  sequence=${dataset_name%_euroc}

  for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    run_name="run${repeat}"
    run_dir="${RESULTS_ROOT}/${sequence}/${run_name}"
    log_path="${run_dir}/run.log"

    if [[ -d ${run_dir} ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
      printf '[skip] %s/%s: output directory is nonempty: %s\n' \
        "${sequence}" "${run_name}" "${run_dir}"
      ((skipped += 1))
      continue
    fi

    printf '[run] %s/%s\n' "${sequence}" "${run_name}"
    printf '  dataset: %s\n  output:  %s\n' "${dataset}" "${run_dir}"
    ((planned += 1))

    if ${DRY_RUN}; then
      printf '  command: '
      printf '%q ' "${BINARY}" "${CONFIG}" "${dataset}/" "${run_dir}/"
      printf '2>&1 | tee %q\n' "${log_path}"
      continue
    fi

    mkdir -p -- "${run_dir}"
    if "${BINARY}" "${CONFIG}" "${dataset}/" "${run_dir}/" \
        2>&1 | tee "${log_path}"; then
      printf '[ok] %s/%s\n' "${sequence}" "${run_name}"
      ((completed += 1))
    else
      status=${PIPESTATUS[0]}
      printf '[failed] %s/%s (OKVIS exit status %d)\n' \
        "${sequence}" "${run_name}" "${status}" >&2
      failed_runs+=("${sequence}/${run_name}:${status}")
      ((failed += 1))
    fi
  done
done

if ${DRY_RUN}; then
  printf 'Dry run complete: pending=%d, existing_skipped=%d. No outputs created.\n' \
    "${planned}" "${skipped}"
  exit 0
fi

printf 'Summary: completed=%d, existing_skipped=%d, failed=%d\n' \
  "${completed}" "${skipped}" "${failed}"
if ((failed > 0)); then
  printf 'Failed runs:\n' >&2
  printf '  %s\n' "${failed_runs[@]}" >&2
  exit 1
fi
