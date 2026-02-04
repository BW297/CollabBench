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
: "${CWAH_AGENT_API_BASE:=https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/717dede2-a628-4ab4-ac35-aaa97b58f255/1b725f6d-8a89-4911-8811-38fcc6078c74/proxy/8042/v1}"
: "${CWAH_AGENT_LM_ID:=deepseek}"
: "${CWAH_AGENT_API_KEY:=EMPTY}"

: "${CWAH_HUMAN_API_BASE:=https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/717dede2-a628-4ab4-ac35-aaa97b58f255/1b725f6d-8a89-4911-8811-38fcc6078c74/proxy/8042/v1}"
: "${CWAH_HUMAN_LM_ID:=deepseek}"
: "${CWAH_HUMAN_API_KEY:=EMPTY}"

export CWAH_AGENT_API_BASE CWAH_AGENT_LM_ID CWAH_AGENT_API_KEY
export CWAH_HUMAN_API_BASE CWAH_HUMAN_LM_ID CWAH_HUMAN_API_KEY

# ===== Sweep Config (matches symbolic_obs_llm_llm_detail.sh behavior) =====
PORT="${PORT:-6391}"
TASKS_STR="${TASKS_STR:-"0 5 10 16 20 26 30 32 40 49"}"
PERSONALITY_START="${PERSONALITY_START:-0}"
PERSONALITY_END="${PERSONALITY_END:-29}" # inclusive

NUM_RUNS="${NUM_RUNS:-1}"
NUM_PER_TASK="${NUM_PER_TASK:-1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMP="${TEMP:-1.0}"

EXECUTABLE_FILE="${EXECUTABLE_FILE:-${ROOT}/../executable/linux_exec.v2.3.0.x86_64}"
DATASET_PATH="${DATASET_PATH:-${ROOT}/dataset/test_env_set_help.pik}"
PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-LLM/prompt_detail.csv}"

# For folder naming only
MODEL_TAG="${MODEL_TAG:-${CWAH_AGENT_LM_ID}}"
# Save runs inside this experiment folder by default
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"

DEBUG_FLAG="${DEBUG_FLAG:---debug}"

cd "${ROOT}"

read -r -a TASKS <<< "${TASKS_STR}"

echo "[sweep] setting_dir=${ROOT}"
echo "[sweep] port=${PORT}"
echo "[sweep] tasks=${TASKS_STR}"
echo "[sweep] personalities=${PERSONALITY_START}..${PERSONALITY_END}"
echo "[sweep] agent: ${CWAH_AGENT_LM_ID} @ ${CWAH_AGENT_API_BASE}"
echo "[sweep] human: ${CWAH_HUMAN_LM_ID} @ ${CWAH_HUMAN_API_BASE}"
echo "[sweep] output_root=${RUNS_ROOT}"
echo "[sweep] num_runs=${NUM_RUNS} num_per_task=${NUM_PER_TASK} temp=${TEMP} max_tokens=${MAX_TOKENS}"
echo "[sweep] tasks_str=${TASKS_STR} personalities=${PERSONALITY_START}..${PERSONALITY_END}"

for task in "${TASKS[@]}"; do
  echo "=============================="
  echo " Running test_task: ${task} "
  echo "=============================="

  for ((personality=PERSONALITY_START; personality<=PERSONALITY_END; personality++)); do
    MODE="LLMs_act_${MODEL_TAG}_${personality}_task${task}"
    RECORD_DIR="${RUNS_ROOT}/${MODE}"
    mkdir -p "${RECORD_DIR}"
    LOG_FILE="${RECORD_DIR}/console.txt"

    {
      echo "---- Running personality: ${personality} (task=${task}) ----"
      echo "[run] record_dir=${RECORD_DIR}"
      echo "[run] logging_to=${LOG_FILE}"
      echo "[run] sweep: port=${PORT} tasks_str=${TASKS_STR} personalities=${PERSONALITY_START}..${PERSONALITY_END} num_runs=${NUM_RUNS} num_per_task=${NUM_PER_TASK} temp=${TEMP} max_tokens=${MAX_TOKENS}"
      echo "[run] llm: agent=${CWAH_AGENT_LM_ID} human=${CWAH_HUMAN_LM_ID}"
      echo "[run] start_time=$(date -Is)"

      if command -v lsof >/dev/null 2>&1; then
        pid="$(lsof -t -i :"${PORT}" || true)"
        if [[ -n "${pid}" ]]; then
          echo "[run] killing pid(s) on :${PORT}: ${pid}"
          kill -9 ${pid} || true
          sleep 0.5
        fi
      fi

      # --- 这里已修正为使用 HUMAN 的配置 ---
      python3 testing_agents/test_symbolic_LLMs.py \
        --communication \
        --prompt_template_path "${PROMPT_TEMPLATE_PATH}" \
        --mode "${MODE}" \
        --record_dir "${RECORD_DIR}" \
        --dataset_path "${DATASET_PATH}" \
        --executable_file "${EXECUTABLE_FILE}" \
        --base-port "${PORT}" \
        --lm_id "${CWAH_HUMAN_LM_ID}" \
        --source openai \
        --api_base "${CWAH_HUMAN_API_BASE}" \
        --t "${TEMP}" \
        --max_tokens "${MAX_TOKENS}" \
        --num_runs "${NUM_RUNS}" \
        --test_task "${task}" \
        --num-per-task "${NUM_PER_TASK}" \
        --cot \
        --act \
        --big_5 "${personality}" \
        ${DEBUG_FLAG}

      echo "[run] end_time=$(date -Is)"
    } 2>&1 | tee -a "${LOG_FILE}"
  done
done
