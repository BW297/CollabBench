#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Sweep helpfulness coefficients (a,b) in [0.6, 1.0] with step=0.1 (5x5=25 settings).
# For each (a,b), this script:
#   1) Re-aggregates trajectory + model_summary from existing `results_lite_obj.jsonl` (NO judge API calls)
#   2) Exports paper tables (raw + norm) for BOTH environments:
#        - coela_11/CoELA/evaluation (out_cwah-*)
#        - coela_11/CoELA/evaluation/proagent (out_src-*)
#   3) Writes outputs into per-setting folders under OUT_ROOT
#
# You can change defaults by exporting env vars before running, e.g.:
#   OUT_ROOT=coela_11/CoELA/sweeps/help_ab GLOBAL_A=1.0 GLOBAL_B=1.0 bash coela_11/CoELA/evaluation/sweep_helpfulness_ab.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # coela_11/CoELA
EVAL_DIR="${ROOT_DIR}/evaluation"
EVALP_DIR="${ROOT_DIR}/evaluation/proagent"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/sweeps/helpfulness_ab}"

mkdir -p "${EVALP_DIR}"

# Default coefficients for non-helpfulness dims (trust/empathy). Leave as-is unless you want to sweep them too.
GLOBAL_A="${GLOBAL_A:-1.0}"
GLOBAL_B="${GLOBAL_B:-1.0}"

# Apply a trajectory-level penalty (max 1pt) when send_message ratio is below threshold.
MSG_PENALTY_THRESHOLD="${MSG_PENALTY_THRESHOLD:-0.2}"
MSG_PENALTY_MAX="${MSG_PENALTY_MAX:-1.0}"
MSG_PENALTY_K="${MSG_PENALTY_K:-4.0}"

DEFAULT_HELP_VALUES=(0.6 0.7 0.8 0.9 1.0)

# Optional override (comma or space separated), e.g.:
#   HELP_A_VALUES="0.6,0.8,1.0"
#   HELP_B_VALUES="0.6 0.8 1.0"
HELP_A_VALUES_ARR=()
HELP_B_VALUES_ARR=()
if [[ -n "${HELP_A_VALUES:-}" ]]; then
  tmp="${HELP_A_VALUES//,/ }"
  IFS=' ' read -r -a HELP_A_VALUES_ARR <<< "${tmp}"
else
  HELP_A_VALUES_ARR=("${DEFAULT_HELP_VALUES[@]}")
fi
if [[ -n "${HELP_B_VALUES:-}" ]]; then
  tmp="${HELP_B_VALUES//,/ }"
  IFS=' ' read -r -a HELP_B_VALUES_ARR <<< "${tmp}"
else
  HELP_B_VALUES_ARR=("${DEFAULT_HELP_VALUES[@]}")
fi

DRY_RUN="${DRY_RUN:-false}"

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

mkdir -p "${OUT_ROOT}"
echo "[info] OUT_ROOT=${OUT_ROOT}"
echo "[info] GLOBAL_A=${GLOBAL_A} GLOBAL_B=${GLOBAL_B}"
echo "[info] MSG_PENALTY_THRESHOLD=${MSG_PENALTY_THRESHOLD} MAX=${MSG_PENALTY_MAX} K=${MSG_PENALTY_K}"
echo "[info] HELP_A_VALUES=$(printf '%s ' "${HELP_A_VALUES_ARR[@]}")"
echo "[info] HELP_B_VALUES=$(printf '%s ' "${HELP_B_VALUES_ARR[@]}")"

for a in "${HELP_A_VALUES_ARR[@]}"; do
  for b in "${HELP_B_VALUES_ARR[@]}"; do
    tag="help_a${a}_b${b}"
    out_dir="${OUT_ROOT}/${tag}"
    mkdir -p "${out_dir}"

    echo
    echo "[sweep] ${tag}"

    # Resume support: skip if all expected outputs already exist.
    exp_files=(
      "${out_dir}/evaluate_proagent_final_paper_table_raw.csv"
      "${out_dir}/evaluate_proagent_final_paper_table_norm.csv"
      "${out_dir}/evaluation_final_paper_table_raw.csv"
      "${out_dir}/evaluation_final_paper_table_norm.csv"
    )
    all_done=true
    for f in "${exp_files[@]}"; do
      if [[ ! -s "${f}" ]]; then
        all_done=false
        break
      fi
    done
    if [[ "${all_done}" == "true" ]]; then
      echo "[skip] ${tag} (outputs already exist)"
      continue
    fi

    # Re-aggregate in-place (fast; overwrites trajectory/model_summary in out_* folders).
    run_cmd python "${EVAL_DIR}/tools/llmjudge_reaggregate_outdirs.py" \
      --base "${EVALP_DIR}" --glob 'out_src-*' \
      --a "${GLOBAL_A}" --b "${GLOBAL_B}" \
      --a-helpfulness "${a}" --b-helpfulness "${b}" \
      --msg-penalty-threshold "${MSG_PENALTY_THRESHOLD}" \
      --msg-penalty-max "${MSG_PENALTY_MAX}" \
      --msg-penalty-k "${MSG_PENALTY_K}" \
      --fast

    run_cmd python "${EVAL_DIR}/tools/llmjudge_reaggregate_outdirs.py" \
      --base "${EVAL_DIR}" --glob 'out_cwah-*' \
      --a "${GLOBAL_A}" --b "${GLOBAL_B}" \
      --a-helpfulness "${a}" --b-helpfulness "${b}" \
      --msg-penalty-threshold "${MSG_PENALTY_THRESHOLD}" \
      --msg-penalty-max "${MSG_PENALTY_MAX}" \
      --msg-penalty-k "${MSG_PENALTY_K}" \
      --fast

    # Export tables into the per-setting folder.
    # ProAgent paper table export can be slow; allow skipping via SKIP_PROAGENT_TABLE=true.
    if [[ "${SKIP_PROAGENT_TABLE:-false}" != "true" ]]; then
      run_cmd python "${EVAL_DIR}/tools/llmjudge_export_paper_table.py" \
        --base "${EVALP_DIR}" --glob 'out_src-*' \
        --out-csv-raw "${out_dir}/evaluate_proagent_final_paper_table_raw.csv" \
        --out-csv-norm "${out_dir}/evaluate_proagent_final_paper_table_norm.csv"
    fi

    run_cmd python "${EVAL_DIR}/tools/llmjudge_export_paper_table.py" \
      --base "${EVAL_DIR}" --glob 'out_cwah-*' \
      --out-csv-raw "${out_dir}/evaluation_final_paper_table_raw.csv" \
      --out-csv-norm "${out_dir}/evaluation_final_paper_table_norm.csv"

    echo "[ok] ${out_dir}"
  done
done

echo
echo "[done] wrote 25 folders under: ${OUT_ROOT}"
