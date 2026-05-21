#!/usr/bin/env bash
set -euo pipefail

LOG_PIK="${1:-}"
AGENT_INDEX="${2:-0}"
STEP_INDEX="${3:-0}"

if [[ -z "${LOG_PIK}" ]]; then
  echo "Usage: $0 <logs_agent_*.pik> [agent_index=0|1] [step_index]"
  exit 2
fi

python3 - "${LOG_PIK}" "${AGENT_INDEX}" "${STEP_INDEX}" <<'PY'
import pickle, sys
path = sys.argv[1]
agent_index = int(sys.argv[2])
step_index = int(sys.argv[3])

data = pickle.load(open(path, "rb"))
entry = data["LLM"][agent_index][step_index]

print("=== raw_input.prompt ===")
print(entry.get("raw_input", {}).get("prompt", ""))
print("\n=== raw_output.text ===")
print(entry.get("raw_output", {}).get("text", ""))
print("\n=== meta ===")
print(entry.get("raw_output", {}).get("meta", {}))
PY
