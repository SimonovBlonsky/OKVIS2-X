#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

CONDA_ENV="okvis2x"
DATA_ROOT="/home/chenguyuan/data"
REFERENCE_RESULTS_ROOT="${REPOSITORY}/workspace/ego2_results"
RESULTS_ROOT="${REFERENCE_RESULTS_ROOT}/202608_causal_diagnostics"
BINARY="/home/chenguyuan/code/okvis_ws/build/okvis/okvis_app_synchronous"
CONFIG="${REPOSITORY}/config/okvis2_eucm_EGO2.yaml"
REPEATS=2
JOBS=1
DRY_RUN=false

usage() {
  cat <<'EOF'
Run unfinished OKVIS causal diagnostics for all 20260803-20260806 sequences.

Usage:
  run_remaining_vio_diagnostics_20260803_20260806.sh [options]

Options:
  --dry-run                       Validate and print commands without running OKVIS
  --conda-env NAME                Conda environment (default: okvis2x)
  --data-root PATH                Dataset root (default: /home/chenguyuan/data)
  --reference-results-root PATH   Existing result root used to discover sequences
  --results-root PATH             Causal diagnostic output root
  --binary PATH                   okvis_app_synchronous binary
  --config PATH                   OKVIS configuration file
  --repeats N                     Repeats per sequence (default: 2)
  --jobs N                        Runner job count (currently only 1 is supported)
  -h, --help                      Show this help

Valid completed run directories are skipped. Partial or invalid output directories
are rejected and are never overwritten.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

require_value() {
  [[ $# -ge 2 ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --conda-env)
      require_value "$@"
      CONDA_ENV=$2
      shift 2
      ;;
    --data-root)
      require_value "$@"
      DATA_ROOT=$2
      shift 2
      ;;
    --reference-results-root)
      require_value "$@"
      REFERENCE_RESULTS_ROOT=$2
      shift 2
      ;;
    --results-root)
      require_value "$@"
      RESULTS_ROOT=$2
      shift 2
      ;;
    --binary)
      require_value "$@"
      BINARY=$2
      shift 2
      ;;
    --config)
      require_value "$@"
      CONFIG=$2
      shift 2
      ;;
    --repeats)
      require_value "$@"
      REPEATS=$2
      shift 2
      ;;
    --jobs)
      require_value "$@"
      JOBS=$2
      shift 2
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

[[ ${REPEATS} =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer"
[[ ${JOBS} =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ -d ${DATA_ROOT} ]] || die "missing data root: ${DATA_ROOT}"
[[ -d ${REFERENCE_RESULTS_ROOT} ]] || \
  die "missing reference result root: ${REFERENCE_RESULTS_ROOT}"
[[ -f ${BINARY} ]] || die "missing OKVIS binary: ${BINARY}"
[[ -f ${CONFIG} ]] || die "missing OKVIS config: ${CONFIG}"
command -v conda >/dev/null 2>&1 || die "conda is not available on PATH"

RUNNER="${REPOSITORY}/tools/accuracy_analysis/scripts/run_vio_diagnostics.py"
[[ -f ${RUNNER} ]] || die "missing diagnostic runner: ${RUNNER}"

declare -a discovered_sequences=()
declare -a skipped_sequences=()
declare -a pending_sequences=()
declare -A dataset_by_sequence=()

for day in 20260803 20260804 20260805 20260806; do
  reference_day="${REFERENCE_RESULTS_ROOT}/${day}"
  [[ -d ${reference_day} ]] || die "missing reference day: ${reference_day}"

  mapfile -t day_sequences < <(
    find "${reference_day}" -type f -name 'mocap*.log' -printf '%h\n' \
      | xargs -r -n1 basename \
      | awk -v prefix="${day}-" 'index($0, prefix) == 1' \
      | sort -u
  )
  ((${#day_sequences[@]} > 0)) || die "no sequences found for ${day}"

  for sequence in "${day_sequences[@]}"; do
    mapfile -t datasets < <(
      find "${DATA_ROOT}/${day}" -type d -name "${sequence}_euroc" -print | sort
    )
    ((${#datasets[@]} == 1)) || \
      die "${sequence}: expected one dataset, found ${#datasets[@]}"
    [[ ${datasets[0]} != *','* && ${datasets[0]} != *$'\n'* ]] || \
      die "dataset path cannot contain a comma or newline: ${datasets[0]}"

    discovered_sequences+=("${sequence}")
    dataset_by_sequence["${sequence}"]=${datasets[0]}

    complete=true
    for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
      sentinel="${RESULTS_ROOT}/${sequence}/run${repeat}/diagnostics/.vio_diagnostics.complete"
      if [[ ! -f ${sentinel} ]]; then
        complete=false
        break
      fi
    done
    if ${complete}; then
      skipped_sequences+=("${sequence}")
    else
      pending_sequences+=("${sequence}")
    fi
  done
done

print_sequences() {
  local title=$1
  shift
  printf '%s (%d):\n' "${title}" "$#"
  if (($# == 0)); then
    printf '  (none)\n'
    return
  fi
  printf '  %s\n' "$@"
}

print_sequences "Discovered sequences" "${discovered_sequences[@]}"
print_sequences "Completed and skipped" "${skipped_sequences[@]}"
print_sequences "Pending sequences" "${pending_sequences[@]}"

if ((${#pending_sequences[@]} == 0)); then
  printf 'All requested diagnostics are already complete.\n'
  exit 0
fi

DATASET_MANIFEST=$(mktemp --tmpdir "okvis-vio-datasets.XXXXXX.csv")
cleanup() {
  rm -f -- "${DATASET_MANIFEST}"
}
trap cleanup EXIT

printf '%s\n' \
  'experiment_id,dataset,sequence,intervention,intervention_value' \
  >"${DATASET_MANIFEST}"
for sequence in "${pending_sequences[@]}"; do
  printf '%s,%s,%s,%s,%s\n' \
    "${sequence}" \
    "${dataset_by_sequence[${sequence}]}" \
    "${sequence}" \
    'baseline' \
    'none' \
    >>"${DATASET_MANIFEST}"
done

command=(
  conda run -n "${CONDA_ENV}" python "${RUNNER}"
  --binary "${BINARY}"
  --config "${CONFIG}"
  --reference-results-root "${REFERENCE_RESULTS_ROOT}"
  --results-root "${RESULTS_ROOT}"
  --dataset-manifest "${DATASET_MANIFEST}"
  --repeats "${REPEATS}"
  --jobs "${JOBS}"
  --skip-complete
)
if ${DRY_RUN}; then
  command+=(--dry-run)
fi

"${command[@]}"
