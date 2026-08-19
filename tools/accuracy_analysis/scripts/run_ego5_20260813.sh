#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

BINARY=${BINARY:-${REPOSITORY}/../../build/okvis/okvis_app_synchronous}
CONFIG=${CONFIG:-${REPOSITORY}/config/okvis2_eucm_EGO5.yaml}
DATA_ROOT=${DATA_ROOT:-/home/chenguyuan/data/20260813}
RESULTS_ROOT=${RESULTS_ROOT:-${REPOSITORY}/workspace/ego5_results/20260813}
REPEATS=${REPEATS:-2}
DRY_RUN=${DRY_RUN:-false}

usage() {
  cat <<'EOF'
Run every EuRoC sequence in /home/chenguyuan/data/20260813 twice.

Usage:
  run_ego5_20260813.sh [--dry-run]

Options:
  --dry-run   Print the runs without starting OKVIS
  -h, --help  Show this help

Existing nonempty run directories are skipped to avoid overwriting results.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
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

mapfile -d '' -t datasets < <(
  find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '*_euroc' \
    -print0 | sort -z
)
((${#datasets[@]} > 0)) || die "no *_euroc datasets found in ${DATA_ROOT}"

printf 'Found %d sequence(s); each sequence will run %d times.\n' \
  "${#datasets[@]}" "${REPEATS}"

completed=0
skipped=0
failed=0
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

    if ${DRY_RUN}; then
      printf '  command: '
      printf '%q ' "${BINARY}" "${CONFIG}" "${dataset}" "${run_dir}/"
      printf '2>&1 | tee %q\n' "${log_path}"
      continue
    fi

    mkdir -p -- "${run_dir}"
    if "${BINARY}" "${CONFIG}" "${dataset}" "${run_dir}/" \
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
  printf 'Dry run complete. No output directories were created.\n'
  exit 0
fi

printf 'Summary: completed=%d, skipped=%d, failed=%d\n' \
  "${completed}" "${skipped}" "${failed}"
if ((failed > 0)); then
  printf 'Failed runs:\n' >&2
  printf '  %s\n' "${failed_runs[@]}" >&2
  exit 1
fi
