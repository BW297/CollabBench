#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${DIR}/.." && pwd)"

ENV_FILE="${ENV_FILE:-}"
if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
elif [[ -f "${DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${DIR}/env.sh"
elif [[ -f "${DIR}/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${DIR}/env.local.sh"
fi

RECORD_DIR="${1:-}"
if [[ -z "${RECORD_DIR}" ]]; then
  echo "Usage: $0 <record_dir> [metric]"
  echo "Example: $0 ../runs/demo_task0_p0 StepNumber"
  exit 2
fi
METRIC="${2:-StepNumber}"

DATASET_PATH="${DATASET_PATH:-${ROOT}/dataset/test_env_set_help.pik}"
NUM_RUNS="${NUM_RUNS:-10}"
NUM_PER_TASK="${NUM_PER_TASK:-2}"

cd "${ROOT}"

python3 utils/statistics.py \
  --record_dir "${RECORD_DIR}" \
  --dataset_path "${DATASET_PATH}" \
  --num_runs "${NUM_RUNS}" \
  --num_per_task "${NUM_PER_TASK}" \
  --generate_result \
  --plot \
  --metric "${METRIC}"
