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

# ===== LLM Config (edit here, or create scripts/env.local.sh) =====
# If you already export these in your shell (or ENV_FILE/env.sh sets them), those values override the defaults below.
: "${CWAH_AGENT_API_BASE:=https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/17cc5f20-18bf-4e4b-a535-7e28505db39a/41a579f9-ac97-46b9-9cc5-1ad2601eed27/proxy/8007/v1}"
: "${CWAH_AGENT_LM_ID:=Qwen2.5-7B-Instruct}"
: "${CWAH_AGENT_API_KEY:=EMPTY}"

: "${CWAH_HUMAN_API_BASE:=https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/34ace8e1-cd75-4fca-a84e-c2ec65a3d5b2/2b9d3482-5b68-4f90-a966-f5a08fd399a0/proxy/8042/v1}"
: "${CWAH_HUMAN_LM_ID:=qwen2.5}"
: "${CWAH_HUMAN_API_KEY:=EMPTY}"

export CWAH_AGENT_API_BASE CWAH_AGENT_LM_ID CWAH_AGENT_API_KEY
export CWAH_HUMAN_API_BASE CWAH_HUMAN_LM_ID CWAH_HUMAN_API_KEY

# Also pass agent-side values via CLI args (as fallback) so configs work even if env vars
# aren't visible in Python subprocesses for some reason.
CLI_LM_ID="${CLI_LM_ID:-${CWAH_AGENT_LM_ID}}"
CLI_API_BASE="${CLI_API_BASE:-${CWAH_AGENT_API_BASE}}"
CLI_API_KEY="${CLI_API_KEY:-${CWAH_AGENT_API_KEY}}"

TASK="${TASK:-0}"
PERSONALITY="${PERSONALITY:-0}"
PORT="${PORT:-6390}"
MODE="${MODE:-demo_task${TASK}_p${PERSONALITY}}"
RECORD_DIR="${RECORD_DIR:-${ROOT}/../runs/${MODE}}"

NUM_RUNS="${NUM_RUNS:-10}"
NUM_PER_TASK="${NUM_PER_TASK:-2}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMP="${TEMP:-1.0}"

EXECUTABLE_FILE="${EXECUTABLE_FILE:-${ROOT}/../executable/linux_exec.v2.3.0.x86_64}"
DATASET_PATH="${DATASET_PATH:-${ROOT}/dataset/test_env_set_help.pik}"
PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-LLM/prompt_detail.csv}"

DEBUG_FLAG="${DEBUG_FLAG:---debug}"

cd "${ROOT}"

echo "[run] setting_dir=${ROOT}"
echo "[run] record_dir=${RECORD_DIR}"
echo "[run] task=${TASK} personality=${PERSONALITY} port=${PORT}"
echo "[run] agent:  ${CWAH_AGENT_LM_ID:-<unset>} @ ${CWAH_AGENT_API_BASE:-<unset>}"
echo "[run] human:  ${CWAH_HUMAN_LM_ID:-<unset>} @ ${CWAH_HUMAN_API_BASE:-<unset>}"

if command -v lsof >/dev/null 2>&1; then
  pid="$(lsof -t -i :"${PORT}" || true)"
  if [[ -n "${pid}" ]]; then
    echo "[run] killing pid(s) on :${PORT}: ${pid}"
    kill -9 ${pid} || true
    sleep 0.5
  fi
fi

python3 testing_agents/test_symbolic_LLMs.py \
  --communication \
  --prompt_template_path "${PROMPT_TEMPLATE_PATH}" \
  --mode "${MODE}" \
  --record_dir "${RECORD_DIR}" \
  --dataset_path "${DATASET_PATH}" \
  --executable_file "${EXECUTABLE_FILE}" \
  --base-port "${PORT}" \
  --lm_id "${CLI_LM_ID}" \
  --source openai \
  --api_base "${CLI_API_BASE}" \
  --api_key "${CLI_API_KEY}" \
  --t "${TEMP}" \
  --max_tokens "${MAX_TOKENS}" \
  --num_runs "${NUM_RUNS}" \
  --test_task "${TASK}" \
  --num-per-task "${NUM_PER_TASK}" \
  --cot \
  --act \
  --big_5 "${PERSONALITY}" \
  ${DEBUG_FLAG}

echo "[done] results: ${RECORD_DIR}/results.pik"
echo "[done] logs:    ${RECORD_DIR}/logs_agent_*"
