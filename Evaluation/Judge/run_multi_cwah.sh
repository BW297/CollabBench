#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          
EVAL_DIR="${ROOT_DIR}/evaluation"                                    

#######################################
# USER CONFIG (optional)
#######################################

# OpenAI-compatible endpoint
# Priority: CLI flags > env > defaults
API_BASE="${OPENAI_API_BASE:-${API_BASE:-http://localhost:8000/v1}}"
API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"   # optional if your server doesn't require it
MODEL="${OPENAI_MODEL:-${MODEL:-deepseek}}"

# Default IDs (used if you pass explicit --agent-id/--human-id or if inference fails)
AGENT_ID="${AGENT_ID:-0}"
HUMAN_ID="${HUMAN_ID:-1}"

# Where to write the final multi-model comparison CSV
COMPARE_CSV="${COMPARE_CSV:-${EVAL_DIR}/model_compare.csv}"

# Optional knobs (can be omitted; pipeline has its own defaults)
WORKERS="${WORKERS:-4}"             # e.g. 4 / 8 / 16
A_WEIGHT="${A_WEIGHT:-1.0}"
B_WEIGHT="${B_WEIGHT:-1.0}"
MAX_FILES="${MAX_FILES:-}"          # e.g. 50
MAX_CHARS="${MAX_CHARS:-2200}"
RESUME="${RESUME:-true}"            # true/false

COMPARE_GLOB="${COMPARE_GLOB:-out_*}"
COMPARE_SORT_BY="${COMPARE_SORT_BY:-overall_of_dimension_means}"  # set to "" to disable sorting
COMPARE_DESCENDING="${COMPARE_DESCENDING:-true}"

#######################################
# 8) TARGETS (choose ONE: VARIANTS or RUNS_DIRS)
#######################################

VARIANTS=(
)

# Option B: list explicit runs directories instead.
# If you use RUNS_DIRS, set VARIANTS=() above (empty).
RUNS_DIRS=()

#######################################
# USER CONFIG END
#######################################

usage() {
  cat <<'EOF'
Usage:
  bash evaluation/run_multi_cwah.sh

Options:
  --api-base URL
  --api-key KEY
  --model NAME
  --agent-id N            (overrides filename inference)
  --human-id N            (overrides filename inference)
  --workers N
  --a FLOAT
  --b FLOAT
  --max-files N
  --max-chars N
  --resume | --no-resume
  --variant PATH          (repeatable; relative to cook_11/cook)
  --runs PATH             (repeatable; explicit runs dir)
  --compare-csv PATH
  --compare-glob GLOB
  --compare-sort-by COL   (empty string disables sorting)
  --compare-desc | --compare-asc
  --dry-run
  -h | --help
EOF
}

die() {
  echo "[error] $*" >&2
  exit 2
}

DRY_RUN=false
CLI_VARIANTS=()
CLI_RUNS=()

# Track whether user explicitly overrides IDs via CLI flags
OVERRIDE_IDS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --agent-id) AGENT_ID="$2"; OVERRIDE_IDS=true; shift 2 ;;
    --human-id) HUMAN_ID="$2"; OVERRIDE_IDS=true; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --a) A_WEIGHT="$2"; shift 2 ;;
    --b) B_WEIGHT="$2"; shift 2 ;;
    --max-files) MAX_FILES="$2"; shift 2 ;;
    --max-chars) MAX_CHARS="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --no-resume) RESUME=false; shift ;;
    --variant) CLI_VARIANTS+=("$2"); shift 2 ;;
    --runs) CLI_RUNS+=("$2"); shift 2 ;;
    --compare-csv) COMPARE_CSV="$2"; shift 2 ;;
    --compare-glob) COMPARE_GLOB="$2"; shift 2 ;;
    --compare-sort-by) COMPARE_SORT_BY="$2"; shift 2 ;;
    --compare-desc) COMPARE_DESCENDING=true; shift ;;
    --compare-asc) COMPARE_DESCENDING=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

if [[ -z "${API_BASE}" ]]; then
  die "API_BASE is empty (pass --api-base or set OPENAI_API_BASE/API_BASE)."
fi
if [[ -z "${MODEL}" ]]; then
  die "MODEL is empty (pass --model or set OPENAI_MODEL/MODEL)."
fi
if [[ -z "${COMPARE_CSV}" ]]; then
  die "COMPARE_CSV is empty (pass --compare-csv or set COMPARE_CSV)."
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

# Infer AGENT_ID/HUMAN_ID from variant folder name.
# Supported patterns:
#   cwah-<H>-human-<A>-agent_...
#   cwah-<A>-agent-<H>-human_...
infer_ids_from_variant() {
  local variant="$1"
  local a="" h=""

  # Pattern: cwah-0-human-1-agent_xxx
  if [[ "${variant}" =~ ^cwah-([0-9]+)-human-([0-9]+)-agent(_|$) ]]; then
    h="${BASH_REMATCH[1]}"
    a="${BASH_REMATCH[2]}"
    echo "${a} ${h}"
    return 0
  fi

  # Pattern: cwah-0-agent-1-human_xxx
  if [[ "${variant}" =~ ^cwah-([0-9]+)-agent-([0-9]+)-human(_|$) ]]; then
    a="${BASH_REMATCH[1]}"
    h="${BASH_REMATCH[2]}"
    echo "${a} ${h}"
    return 0
  fi

  return 1
}

if [[ ${#CLI_VARIANTS[@]} -gt 0 && ${#CLI_RUNS[@]} -gt 0 ]]; then
  die "Provide only one of: --variant ... OR --runs ..."
fi

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] EVAL_DIR=${EVAL_DIR}"
echo "[info] API_BASE=${API_BASE}"
echo "[info] MODEL=${MODEL}"
echo "[info] WORKERS=${WORKERS} A=${A_WEIGHT} B=${B_WEIGHT} RESUME=${RESUME}"
echo "[info] MAX_FILES=${MAX_FILES:-<unset>} MAX_CHARS=${MAX_CHARS}"
echo "[info] COMPARE_CSV=${COMPARE_CSV} COMPARE_GLOB=${COMPARE_GLOB} SORT_BY=${COMPARE_SORT_BY:-<disabled>} DESC=${COMPARE_DESCENDING}"
echo "[info] ID mode: $([[ "${OVERRIDE_IDS}" == "true" ]] && echo "CLI override (AGENT_ID=${AGENT_ID} HUMAN_ID=${HUMAN_ID})" || echo "Infer from variant filename (fallback AGENT_ID=${AGENT_ID} HUMAN_ID=${HUMAN_ID})")"

# Build common args that do NOT include agent/human id (those may vary per variant when inferring)
common_args_base=(
  --api-base "${API_BASE}"
  --model "${MODEL}"
)
if [[ -n "${API_KEY}" ]]; then
  common_args_base+=( --api-key "${API_KEY}" )
fi
if [[ -n "${WORKERS}" ]]; then
  common_args_base+=( --workers "${WORKERS}" )
fi
if [[ -n "${MAX_CHARS}" ]]; then
  common_args_base+=( --max-chars "${MAX_CHARS}" )
fi
if [[ -n "${A_WEIGHT}" ]]; then
  common_args_base+=( --a "${A_WEIGHT}" )
fi
if [[ -n "${B_WEIGHT}" ]]; then
  common_args_base+=( --b "${B_WEIGHT}" )
fi
if [[ -n "${MAX_FILES}" ]]; then
  common_args_base+=( --max-files "${MAX_FILES}" )
fi
if [[ "${RESUME}" == "true" ]]; then
  common_args_base+=( --resume )
fi

# Helper to run a variant with proper IDs
run_variant() {
  local variant="$1"
  local local_agent_id="${AGENT_ID}"
  local local_human_id="${HUMAN_ID}"

  if [[ "${OVERRIDE_IDS}" != "true" ]]; then
    if ids="$(infer_ids_from_variant "${variant}")"; then
      # ids format: "<agent> <human>"
      local_agent_id="$(awk '{print $1}' <<< "${ids}")"
      local_human_id="$(awk '{print $2}' <<< "${ids}")"
      echo "[info] Inferred IDs for ${variant}: AGENT_ID=${local_agent_id} HUMAN_ID=${local_human_id}"
    else
      echo "[warn] Could not infer IDs from variant name: ${variant}. Using fallback AGENT_ID=${local_agent_id} HUMAN_ID=${local_human_id}"
    fi
  fi

  run_cmd python "${EVAL_DIR}/llmjudge.py" run-pipeline \
    --variant "${variant}" \
    --agent-id "${local_agent_id}" \
    --human-id "${local_human_id}" \
    "${common_args_base[@]}"
}

# For runs dirs, we cannot infer from filename reliably; use provided/default IDs unless overridden.
run_runs_dir() {
  local runs_dir="$1"
  run_cmd python "${EVAL_DIR}/llmjudge.py" run-pipeline \
    --runs "${runs_dir}" \
    --agent-id "${AGENT_ID}" \
    --human-id "${HUMAN_ID}" \
    "${common_args_base[@]}"
}

if [[ ${#CLI_RUNS[@]} -gt 0 ]]; then
  echo "[info] Using CLI --runs (${#CLI_RUNS[@]} dirs)"
  for runs_dir in "${CLI_RUNS[@]}"; do
    echo "[run] runs=${runs_dir}"
    run_runs_dir "${runs_dir}"
  done
elif [[ ${#CLI_VARIANTS[@]} -gt 0 ]]; then
  echo "[info] Using CLI --variant (${#CLI_VARIANTS[@]} variants)"
  for variant in "${CLI_VARIANTS[@]}"; do
    echo "[run] variant=${variant}"
    run_variant "${variant}"
  done
elif [[ ${#RUNS_DIRS[@]} -gt 0 ]]; then
  echo "[info] Using RUNS_DIRS (${#RUNS_DIRS[@]} dirs)"
  for runs_dir in "${RUNS_DIRS[@]}"; do
    echo "[run] runs=${runs_dir}"
    run_runs_dir "${runs_dir}"
  done
else
  echo "[info] Using VARIANTS (${#VARIANTS[@]} variants)"
  for variant in "${VARIANTS[@]}"; do
    echo "[run] variant=${variant}"
    run_variant "${variant}"
  done
fi

compare_args=(
  --base "${EVAL_DIR}"
  --glob "${COMPARE_GLOB}"
  --out-csv "${COMPARE_CSV}"
)
if [[ -n "${COMPARE_SORT_BY}" ]]; then
  compare_args+=( --sort-by "${COMPARE_SORT_BY}" )
fi
if [[ "${COMPARE_DESCENDING}" == "true" ]]; then
  compare_args+=( --descending )
fi

echo "[info] Writing comparison CSV: ${COMPARE_CSV}"
run_cmd python "${EVAL_DIR}/llmjudge.py" compare-model-summaries "${compare_args[@]}"

echo "[done]"
