#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

BINARY=${BINARY:-${REPOSITORY}/../../build/okvis/okvis_app_synchronous}
CONFIG=${CONFIG:-${REPOSITORY}/config/okvis2_eucm_EGO0.yaml}
DATA_ROOT=${DATA_ROOT:-/home/chenguyuan/data/20260813_ego0}
RESULTS_ROOT=${RESULTS_ROOT:-${REPOSITORY}/workspace/ego0_results/20260813_rerun_20260817}
REPEATS=${REPEATS:-2}
DRY_RUN=${DRY_RUN:-false}
SEQUENCES=(
  20260813-180940
  20260813-181451
  20260813-182322
  20260813-185137
)
DIAGNOSTIC_FILES=(
  vio_diag_metadata.csv
  vio_diag_frame.csv
  vio_diag_triangulation.csv
  vio_diag_initialisation.csv
  vio_diag_ransac.csv
  vio_diag_landmark_events.csv
)

usage() {
  cat <<'EOF'
Run the selected lower-bitrate EGO0 EuRoC sequences twice with VIO diagnostics.

Usage:
  run_ego0_all_20260813.sh [--dry-run]

Options:
  --dry-run   Print pending OKVIS commands without running them
  -h, --help  Show this help

Results are written to
workspace/ego0_results/20260813_rerun_20260817/<sequence>/run{1,2}.
Existing nonempty run directories are skipped to avoid overwriting results.
Invalid or incomplete EuRoC datasets are reported and skipped.
Diagnostic CSV files are written to each run's diagnostics/ directory.
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

validate_diagnostics() {
  local diagnostics_dir=$1
  local filename

  for filename in "${DIAGNOSTIC_FILES[@]}"; do
    if [[ ! -f ${diagnostics_dir}/${filename} ]]; then
      printf 'missing diagnostic file: %s/%s\n' \
        "${diagnostics_dir}" "${filename}" >&2
      return 1
    fi
  done
  [[ -f ${diagnostics_dir}/.vio_diagnostics.complete ]] || {
    printf 'missing diagnostic completion marker: %s\n' \
      "${diagnostics_dir}/.vio_diagnostics.complete" >&2
    return 1
  }
  [[ ! -e ${diagnostics_dir}/.vio_diagnostics.active ]] || {
    printf 'diagnostic active marker remains: %s\n' \
      "${diagnostics_dir}/.vio_diagnostics.active" >&2
    return 1
  }
  grep -q '^1,run_complete,true$' \
    "${diagnostics_dir}/vio_diag_metadata.csv" || {
    printf 'diagnostic metadata does not contain run_complete=true: %s\n' \
      "${diagnostics_dir}/vio_diag_metadata.csv" >&2
    return 1
  }
  ! grep -q '^1,writer_failed,true$' \
    "${diagnostics_dir}/vio_diag_metadata.csv"
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
[[ -d ${DATA_ROOT} ]] || die "missing data root: ${DATA_ROOT}"

declare -a datasets=()
declare -a invalid_datasets=()
for sequence in "${SEQUENCES[@]}"; do
  dataset="${DATA_ROOT}/${sequence}_euroc"
  if is_complete_euroc "${dataset}"; then
    datasets+=("${dataset}")
  else
    invalid_datasets+=("${sequence}")
    printf '[skip dataset] incomplete or invalid: %s\n' "${dataset}" >&2
  fi
done

((${#datasets[@]} > 0)) || die "none of the selected EuRoC datasets is complete"
printf 'Selected %d sequence(s): complete=%d, invalid=%d; each complete sequence will run %d times with diagnostics.\n' \
  "${#SEQUENCES[@]}" "${#datasets[@]}" "${#invalid_datasets[@]}" \
  "${REPEATS}"

BUILD_ID=$(git -C "${REPOSITORY}" rev-parse --verify HEAD 2>/dev/null || \
  printf 'unknown')
if [[ -n $(git -C "${REPOSITORY}" status --porcelain 2>/dev/null) ]]; then
  BUILD_ID="${BUILD_ID}-dirty"
fi

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
    diagnostics_dir="${run_dir}/diagnostics"

    if [[ -d ${run_dir} ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
      printf '[skip run] %s/%s: output directory is nonempty: %s\n' \
        "${sequence}" "${run_name}" "${run_dir}"
      ((skipped += 1))
      continue
    fi

    printf '[run] %s/%s\n' "${sequence}" "${run_name}"
    printf '  dataset:     %s\n  output:      %s\n  diagnostics: %s\n' \
      "${dataset}" "${run_dir}" "${diagnostics_dir}"
    ((planned += 1))

    if ${DRY_RUN}; then
      printf '  command: '
      printf 'OKVIS_DIAGNOSTICS_DIR=%q ' "${diagnostics_dir}"
      printf 'OKVIS_DIAGNOSTICS_RUN_ID=%q ' "${sequence}-${run_name}"
      printf 'OKVIS_DIAGNOSTICS_BUILD_ID=%q ' "${BUILD_ID}"
      printf 'OKVIS_DIAGNOSTICS_DATASET_ID=%q ' "${sequence}"
      printf 'QT_QPA_PLATFORM=offscreen '
      printf '%q ' "${BINARY}" "${CONFIG}" "${dataset}/" "${run_dir}"
      printf '2>&1 | tee %q\n' "${log_path}"
      continue
    fi

    mkdir -p -- "${run_dir}"
    if OKVIS_DIAGNOSTICS_DIR="${diagnostics_dir}" \
        OKVIS_DIAGNOSTICS_RUN_ID="${sequence}-${run_name}" \
        OKVIS_DIAGNOSTICS_BUILD_ID="${BUILD_ID}" \
        OKVIS_DIAGNOSTICS_DATASET_ID="${sequence}" \
        QT_QPA_PLATFORM=offscreen \
        "${BINARY}" "${CONFIG}" "${dataset}/" "${run_dir}" \
        2>&1 | tee "${log_path}"; then
      if validate_diagnostics "${diagnostics_dir}"; then
        printf '[ok] %s/%s\n' "${sequence}" "${run_name}"
        ((completed += 1))
      else
        printf '[failed] %s/%s: diagnostics are incomplete\n' \
          "${sequence}" "${run_name}" >&2
        failed_runs+=("${sequence}/${run_name}:invalid-diagnostics")
        ((failed += 1))
      fi
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
  printf 'Dry run complete: pending=%d, existing_skipped=%d, invalid_datasets=%d. No outputs created.\n' \
    "${planned}" "${skipped}" "${#invalid_datasets[@]}"
  exit 0
fi

printf 'Summary: completed=%d, existing_skipped=%d, failed=%d, invalid_datasets=%d\n' \
  "${completed}" "${skipped}" "${failed}" "${#invalid_datasets[@]}"
if ((failed > 0)); then
  printf 'Failed runs:\n' >&2
  printf '  %s\n' "${failed_runs[@]}" >&2
fi
((failed == 0 && ${#invalid_datasets[@]} == 0))
