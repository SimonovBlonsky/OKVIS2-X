#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

BINARY=${BINARY:-${REPOSITORY}/../../build/okvis/okvis_app_synchronous}
CONFIG=${CONFIG:-${REPOSITORY}/config/okvis2_eucm_EGO5.yaml}
DATA_ROOT=${DATA_ROOT:-/home/chenguyuan/data}
REFERENCE_RESULTS_ROOT=${REFERENCE_RESULTS_ROOT:-${REPOSITORY}/workspace/ego2_results}
RESULTS_ROOT=${RESULTS_ROOT:-${REPOSITORY}/workspace/ego2_results/20260812}

python3 "${SCRIPT_DIR}/run_vio_diagnostics.py" \
  --binary "${BINARY}" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --reference-results-root "${REFERENCE_RESULTS_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --sequences \
    20260812-173946 \
    20260812-174602 \
    20260812-175223 \
    20260812-175801 \
  --repeats 2 \
  --jobs 1 \
  --skip-complete
