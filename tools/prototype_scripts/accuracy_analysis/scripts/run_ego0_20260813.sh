#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

ROS_SETUP=${ROS_SETUP:-/opt/ros/humble/setup.bash}
CONVERTER="${REPOSITORY}/tools/mcap_vio_to_euroc.py"
BINARY=${BINARY:-${REPOSITORY}/../../build/okvis/okvis_app_synchronous}
CONFIG=${CONFIG:-${REPOSITORY}/config/okvis2_eucm_EGO0.yaml}
DATA_ROOT=${DATA_ROOT:-/home/chenguyuan/data/20260813_ego0}
RESULTS_ROOT=${RESULTS_ROOT:-${REPOSITORY}/workspace/ego0_results/20260813}
REPEATS=${REPEATS:-2}
DRY_RUN=${DRY_RUN:-false}

usage() {
  cat <<'EOF'
Convert all EGO0 MCAP sequences from 20260813 to EuRoC and run OKVIS twice.

Usage:
  run_ego0_20260813.sh [--dry-run]

Options:
  --dry-run   Print conversion and OKVIS commands without running them
  -h, --help  Show this help

Converted datasets are written beside their source directories as
<sequence>_euroc. Results are written to
workspace/ego0_results/20260813/<sequence>/run{1,2}.

Complete conversions and nonempty run directories are skipped. Sequences with
failed, incomplete, or invalid conversions are reported and excluded from the
OKVIS stage; their partial outputs are never overwritten.
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

[[ -f ${ROS_SETUP} ]] || die "missing ROS 2 setup: ${ROS_SETUP}"
[[ -f ${CONVERTER} ]] || die "missing converter: ${CONVERTER}"
[[ -x ${BINARY} ]] || die "missing executable: ${BINARY}"
[[ -f ${CONFIG} ]] || die "missing config: ${CONFIG}"
[[ -d ${DATA_ROOT} ]] || die "missing data root: ${DATA_ROOT}"

set +u
# The ROS setup scripts may inspect variables that are not defined yet.
source "${ROS_SETUP}"
set -u

if ! ${DRY_RUN}; then
  python3 -c \
    'import cv2, numpy, rosbag2_py; from rclpy.serialization import deserialize_message; from sensor_msgs.msg import CompressedImage, Imu' \
    || die "missing converter dependencies after sourcing ${ROS_SETUP}"
fi

mapfile -d '' -t source_datasets < <(
  find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name '*_euroc' \
    -exec test -f '{}/cam0/metadata.yaml' ';' -print0 | sort -z
)
((${#source_datasets[@]} > 0)) || \
  die "no source MCAP sequences found in ${DATA_ROOT}"

printf 'Found %d source sequence(s).\n' "${#source_datasets[@]}"
printf 'Stage 1/2: convert MCAP sequences to EuRoC.\n'

converted=0
conversion_reused=0
conversion_failed=0
declare -a runnable_sequences=()
declare -a failed_conversions=()
for source_dataset in "${source_datasets[@]}"; do
  sequence=${source_dataset##*/}
  euroc_dataset="${DATA_ROOT}/${sequence}_euroc"
  incomplete_dataset="${euroc_dataset}.incomplete"

  if [[ -e ${incomplete_dataset} ]]; then
    printf '[skip conversion] %s: incomplete output exists: %s\n' \
      "${sequence}" "${incomplete_dataset}" >&2
    failed_conversions+=("${sequence}:incomplete-output")
    ((conversion_failed += 1))
    continue
  fi
  if [[ -e ${euroc_dataset} ]]; then
    if ! is_complete_euroc "${euroc_dataset}"; then
      printf '[skip conversion] %s: converted dataset is invalid: %s\n' \
        "${sequence}" "${euroc_dataset}" >&2
      failed_conversions+=("${sequence}:invalid-output")
      ((conversion_failed += 1))
      continue
    fi
    printf '[skip conversion] %s: complete output exists\n' "${sequence}"
    runnable_sequences+=("${sequence}")
    ((conversion_reused += 1))
    continue
  fi

  printf '[convert] %s\n' "${sequence}"
  if ${DRY_RUN}; then
    printf '  command: '
    printf '%q ' python3 "${CONVERTER}" "${source_dataset}" "${euroc_dataset}"
    printf '\n'
    runnable_sequences+=("${sequence}")
    continue
  fi

  if ! python3 "${CONVERTER}" "${source_dataset}" "${euroc_dataset}"; then
    printf '[skip sequence] %s: conversion command failed\n' "${sequence}" >&2
    failed_conversions+=("${sequence}:converter-failed")
    ((conversion_failed += 1))
    continue
  fi
  if ! is_complete_euroc "${euroc_dataset}"; then
    printf '[skip sequence] %s: conversion output is incomplete\n' \
      "${sequence}" >&2
    failed_conversions+=("${sequence}:incomplete-output")
    ((conversion_failed += 1))
    continue
  fi
  runnable_sequences+=("${sequence}")
  ((converted += 1))
done

printf 'Stage 2/2: run each EuRoC sequence %d times.\n' "${REPEATS}"
printf 'Runnable sequences: %d; skipped after conversion: %d.\n' \
  "${#runnable_sequences[@]}" "${conversion_failed}"

completed=0
run_skipped=0
failed=0
declare -a failed_runs=()

for sequence in "${runnable_sequences[@]}"; do
  euroc_dataset="${DATA_ROOT}/${sequence}_euroc"

  for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    run_name="run${repeat}"
    run_dir="${RESULTS_ROOT}/${sequence}/${run_name}"
    log_path="${run_dir}/run.log"

    if [[ -d ${run_dir} ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
      printf '[skip run] %s/%s: output directory is nonempty\n' \
        "${sequence}" "${run_name}"
      ((run_skipped += 1))
      continue
    fi

    printf '[run] %s/%s\n' "${sequence}" "${run_name}"
    if ${DRY_RUN}; then
      printf '  command: '
      printf '%q ' "${BINARY}" "${CONFIG}" "${euroc_dataset}" "${run_dir}/"
      printf '2>&1 | tee %q\n' "${log_path}"
      continue
    fi

    mkdir -p -- "${run_dir}"
    if "${BINARY}" "${CONFIG}" "${euroc_dataset}" "${run_dir}/" \
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
  printf 'Dry run complete: runnable sequences=%d, OKVIS runs=%d, skipped conversions=%d. No outputs created.\n' \
    "${#runnable_sequences[@]}" \
    "$((${#runnable_sequences[@]} * REPEATS))" \
    "${conversion_failed}"
  exit 0
fi

printf 'Summary: converted=%d, conversion_reused=%d, conversion_failed=%d, completed_runs=%d, skipped_runs=%d, failed_runs=%d\n' \
  "${converted}" "${conversion_reused}" "${conversion_failed}" \
  "${completed}" "${run_skipped}" "${failed}"
if ((conversion_failed > 0)); then
  printf 'Skipped conversions:\n' >&2
  printf '  %s\n' "${failed_conversions[@]}" >&2
fi
if ((failed > 0)); then
  printf 'Failed runs:\n' >&2
  printf '  %s\n' "${failed_runs[@]}" >&2
fi
((conversion_failed == 0 && failed == 0))
