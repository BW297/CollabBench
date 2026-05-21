#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}"
EVALP_DIR="${SCRIPT_DIR}/cook"

API_BASE="${OPENAI_API_BASE:-${API_BASE:-http://localhost:8000/v1}}"
API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"   # optional if your server doesn't require it
MODEL="${OPENAI_MODEL:-${MODEL:-deepseek}}"

AGENT_ID="${AGENT_ID:-}"          # optional (auto-infer from src-*-agent-*-human-* if empty)
HUMAN_ID="${HUMAN_ID:-}"          # optional
WORKERS="${WORKERS:-4}"
MAX_FILES="${MAX_FILES:-}"        # optional
MAX_CHARS="${MAX_CHARS:-2200}"
A_WEIGHT="${A_WEIGHT:-0.8}"
B_WEIGHT="${B_WEIGHT:-0.8}"
# Optional per-dimension overrides (default: inherit A_WEIGHT/B_WEIGHT)
A_HELPFULNESS="${A_HELPFULNESS:-}"
B_HELPFULNESS="${B_HELPFULNESS:-}"
A_TRUSTFULNESS="${A_TRUSTFULNESS:-}"
B_TRUSTFULNESS="${B_TRUSTFULNESS:-}"
A_EMPATHY="${A_EMPATHY:-}"
B_EMPATHY="${B_EMPATHY:-}"
RESUME="${RESUME:-true}"

COMPARE_CSV="${COMPARE_CSV:-${EVALP_DIR}/model_compare.csv}"
COMPARE_GLOB="${COMPARE_GLOB:-out_*}"
COMPARE_SORT_BY="${COMPARE_SORT_BY:-overall_of_dimension_means}"
COMPARE_DESCENDING="${COMPARE_DESCENDING:-true}"

# Cleanup big intermediates after each model (recommended to avoid quota issues)
CLEANUP="${CLEANUP:-true}"

mkdir -p "${EVALP_DIR}"

DATA_ROOTS_DEFAULT=(
)
DATA_ROOTS=()

usage() {
  cat <<'EOF'
Usage:
  bash evaluation/run_multi_cook.sh --data-root PATH [--data-root PATH ...] [options]

Required:
  --data-root PATH          cook experiment root (contains actions/**/step_1/actions.json). Repeatable.

Options:
  --api-base URL            (default: OPENAI_API_BASE or http://localhost:8000/v1)
  --api-key KEY             (default: OPENAI_API_KEY or EMPTY)
  --model NAME              (default: OPENAI_MODEL or deepseek)
  --agent-id N              (optional; auto-infer if possible)
  --human-id N              (optional)
  --workers N               (default: 4)
  --max-files N             (optional; for quick testing)
  --max-chars N             (default: 2200)
  --a FLOAT                 (default: 1.0)
  --b FLOAT                 (default: 1.0)
  --a-helpfulness FLOAT     (optional; default: --a)
  --b-helpfulness FLOAT     (optional; default: --b)
  --a-trustfulness FLOAT    (optional; default: --a)
  --b-trustfulness FLOAT    (optional; default: --b)
  --a-empathy FLOAT         (optional; default: --a)
  --b-empathy FLOAT         (optional; default: --b)
  --resume | --no-resume    (default: resume)
  --cleanup | --no-cleanup  (default: cleanup)
  --compare-csv PATH        (default: Evaluation/Judge/cook/model_compare.csv)
  --dry-run
  -h | --help

EOF
}

die() { echo "[error] $*" >&2; exit 2; }

DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --data-root) DATA_ROOTS+=("$2"); shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --agent-id) AGENT_ID="$2"; shift 2 ;;
    --human-id) HUMAN_ID="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-files) MAX_FILES="$2"; shift 2 ;;
    --max-chars) MAX_CHARS="$2"; shift 2 ;;
    --a) A_WEIGHT="$2"; shift 2 ;;
    --b) B_WEIGHT="$2"; shift 2 ;;
    --a-helpfulness) A_HELPFULNESS="$2"; shift 2 ;;
    --b-helpfulness) B_HELPFULNESS="$2"; shift 2 ;;
    --a-trustfulness) A_TRUSTFULNESS="$2"; shift 2 ;;
    --b-trustfulness) B_TRUSTFULNESS="$2"; shift 2 ;;
    --a-empathy) A_EMPATHY="$2"; shift 2 ;;
    --b-empathy) B_EMPATHY="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --no-resume) RESUME=false; shift ;;
    --cleanup) CLEANUP=true; shift ;;
    --no-cleanup) CLEANUP=false; shift ;;
    --compare-csv) COMPARE_CSV="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

if [[ ${#DATA_ROOTS[@]} -eq 0 ]]; then
  DATA_ROOTS=( "${DATA_ROOTS_DEFAULT[@]}" )
fi
if [[ -z "${API_BASE}" || -z "${MODEL}" ]]; then
  die "API_BASE/MODEL is empty"
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

common_args=(
  --api-base "${API_BASE}"
  --api-key "${API_KEY}"
  --model "${MODEL}"
  --workers "${WORKERS}"
  --max-chars "${MAX_CHARS}"
  --a "${A_WEIGHT}"
  --b "${B_WEIGHT}"
)
if [[ -n "${A_HELPFULNESS}" ]]; then common_args+=( --a-helpfulness "${A_HELPFULNESS}" ); fi
if [[ -n "${B_HELPFULNESS}" ]]; then common_args+=( --b-helpfulness "${B_HELPFULNESS}" ); fi
if [[ -n "${A_TRUSTFULNESS}" ]]; then common_args+=( --a-trustfulness "${A_TRUSTFULNESS}" ); fi
if [[ -n "${B_TRUSTFULNESS}" ]]; then common_args+=( --b-trustfulness "${B_TRUSTFULNESS}" ); fi
if [[ -n "${A_EMPATHY}" ]]; then common_args+=( --a-empathy "${A_EMPATHY}" ); fi
if [[ -n "${B_EMPATHY}" ]]; then common_args+=( --b-empathy "${B_EMPATHY}" ); fi
if [[ -n "${MAX_FILES}" ]]; then
  common_args+=( --max-files "${MAX_FILES}" )
fi
if [[ -n "${AGENT_ID}" ]]; then
  common_args+=( --agent-id "${AGENT_ID}" )
fi
if [[ -n "${HUMAN_ID}" ]]; then
  common_args+=( --human-id "${HUMAN_ID}" )
fi
if [[ "${RESUME}" == "true" ]]; then
  common_args+=( --resume )
fi
if [[ "${CLEANUP}" == "true" ]]; then
  common_args+=( --cleanup )
fi

echo "[info] API_BASE=${API_BASE}"
echo "[info] MODEL=${MODEL} WORKERS=${WORKERS} MAX_FILES=${MAX_FILES:-<unset>} MAX_CHARS=${MAX_CHARS} A=${A_WEIGHT} B=${B_WEIGHT} (A/B per-dim overrides: help=${A_HELPFULNESS:-<unset>}/${B_HELPFULNESS:-<unset>} trust=${A_TRUSTFULNESS:-<unset>}/${B_TRUSTFULNESS:-<unset>} emp=${A_EMPATHY:-<unset>}/${B_EMPATHY:-<unset>}) RESUME=${RESUME} CLEANUP=${CLEANUP}"

valid_roots=0
for dr in "${DATA_ROOTS[@]}"; do
  dr_norm="${dr%/}"
  if [[ "${dr_norm##*/}" == "actions" ]]; then
    dr_norm="${dr_norm%/actions}"
  fi
  if [[ ! -d "${dr_norm}" ]]; then
    echo "[warn] skip missing data-root: ${dr_norm}"
    continue
  fi
  valid_roots=$((valid_roots + 1))
  echo "[run] data-root=${dr_norm}"
  run_cmd python "${EVAL_DIR}/cook.py" run-pipeline --data-root "${dr_norm}" "${common_args[@]}"
done

if [[ "${valid_roots}" -eq 0 ]]; then
  die "No valid Cook data roots found. Pass at least one existing --data-root PATH."
fi

echo "[info] Writing comparison CSV: ${COMPARE_CSV}"
compare_args=( --base "${EVALP_DIR}" --glob "${COMPARE_GLOB}" --out-csv "${COMPARE_CSV}" --sort-by "${COMPARE_SORT_BY}" )
if [[ "${COMPARE_DESCENDING}" == "true" ]]; then
  compare_args+=( --descending )
fi
run_cmd python "${EVAL_DIR}/llmjudge.py" compare-model-summaries "${compare_args[@]}"

echo "[done]"
