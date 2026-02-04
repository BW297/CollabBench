#!/usr/bin/env bash
set -euo pipefail

# Copy this file to env.local.sh and edit values:
#   cp scripts/env.example.sh scripts/env.local.sh
# Then run:
#   ./scripts/run_one_llm_llm.sh

# ---- LLM runtime (2-player) ----
# Agent side (the "Agent端" in your paper)
export CWAH_AGENT_API_BASE="http://127.0.0.1:8007/v1"
export CWAH_AGENT_LM_ID="qwen2.5-72b-instruct"
# export CWAH_AGENT_API_KEY="sk-xxxx"   # optional; if your server checks keys

# Human side (the fixed "Human端", e.g. deepseekV3.1)
export CWAH_HUMAN_API_BASE="http://127.0.0.1:8042/v1"
export CWAH_HUMAN_LM_ID="deepseek-v3.1"
# export CWAH_HUMAN_API_KEY="sk-xxxx"   # optional

# ---- Judge (subjective eval) ----
export CWAH_JUDGE_API_BASE="${CWAH_HUMAN_API_BASE}"
export CWAH_JUDGE_MODEL="ds-v3.1"
# export CWAH_JUDGE_API_KEY="EMPTY"
export CWAH_JUDGE_MAX_TOKENS="2048"

# ---- Run defaults (optional) ----
export PORT="6390"
export NUM_RUNS="10"
export NUM_PER_TASK="2"
export MAX_TOKENS="4096"
export TEMP="1.0"

