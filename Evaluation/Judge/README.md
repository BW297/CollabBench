# LLM Judge (High-level Action Windows)

This directory provides the LLM-Judge pipeline (intermediate files can be saved):
1) Build windows: `windows.jsonl` + optional `preview.txt`
2) Build prompts from windows: `prompts.jsonl`
3) Score prompts: `results.jsonl`
4) Aggregate window results to trajectory scores: `trajectory.jsonl`
5) Export window-level CSV: `window_scores.csv`
6) Merge outputs: detailed + compact results
7) Model-level summary: mean over trajectories

Windowing rules:
- 1 window = 3 high-level actions (the tail window with <3 actions is dropped)
- Each window is evaluated by 3 prompts and yields 3 scores: `Helpfulness / Trustfulness / Empathy`
- Scoring is penalty-based: max 5, outputs violation points and evidence


## Inputs
Supported inputs: (Take `CWAH-MultiPlayer` as an example)
- A single `logs_agent_*.pik`
- A single run directory (containing `logs_agent_*.pik`)
- A runs root (auto-searches `*/logs_agent_*.pik`)

## Outputs
- `llmjudge.py build-windows --out`: windows JSONL (one window per line, includes `window_text`)
- `llmjudge.py build-windows --preview-out`: human-readable preview text
- `llmjudge.py build-prompts --out`: prompts JSONL (one request per line: one window + one dimension)
- `llmjudge.py score-prompts --out`: results JSONL (one scored request per line)
- `llmjudge.py score-prompts --lite`: results JSONL (drops large fields, smaller for archiving)
- `llmjudge.py aggregate-results --out`: trajectory JSONL (one score per trajectory per dimension)
- `llmjudge.py export-window-csv --out-csv`: window_scores.csv (one row per window, 3 dimension scores)
- `llmjudge.py merge-outputs`:
  - `--out-detailed`: detailed results (one row per window, includes window_text + full results)
  - `--out-compact`: compact results (one row per window, only scores + rule IDs)
- `llmjudge.py run-pipeline`: one-shot full pipeline (intermediates + detailed/compact + CSV + trajectory)
- `llmjudge.py model-summary`: summarize `trajectory.jsonl` into model-level means
- `llmjudge.py compare-model-summaries`: combine multiple outdirs into a comparison CSV

## Objective fields
Objective metrics extracted from logs are stored in the `objective` field for analysis:
- `total_env_steps`: total env steps (prefer agent action length; fallback to progress/goals_finished length)
- `agent_env_steps` / `human_env_steps`: env action steps (`len(log["action"][id])`)
- `agent_send_message_steps` / `human_send_message_steps`: count of `[send_message]`
- `agent_llm_calls` / `human_llm_calls`: LLM call counts (`len(log["LLM"][id])`)
- `finished` / `task_success`: completion flags if present

## Re-run workflow (recommended)
Each step can be re-run independently; outputs are chained:
- Step1 output `windows.jsonl` is Step2/3 input
- Step2 output `requests_thin.jsonl` (or `prompts_full.jsonl`) is Step3 input
- Step3 output `results_lite.jsonl` is Step4/5/6/7 input

Recommended layout: keep all files of one experiment in a single `out_dir/` with fixed names:
- `windows.jsonl`, `windows_preview.txt`
- `requests_thin.jsonl` (or `prompts_full.jsonl`)
- `results_lite.jsonl`
- `results_detailed.jsonl`, `results_compact.jsonl`
- `window_scores.csv`
- `trajectory.jsonl`, `trajectory.csv`
- `model_summary.json`, `model_summary.csv`

Common re-run cases:
- Change windowing/context: re-run Step1 (affects all later steps)
- Change prompts/scoring model: keep Step1/2, re-run Step3 (and 4/5/6/7)
- Change a/b weights: re-run Step4 only
- Export analysis CSV: Step5 only
- Produce detailed/compact results: Step6 only
- Model summary: Step7 only

## Cook-MultiPlayer
Cook-MultiPlayer `actions.json` evaluation is handled by `cook.py`. By default, Cook outputs are written under `Evaluation/Judge/cook/out_<data_root_name>/`.

## Examples
Quick multi-run scripts are provided (set API env vars before running):
```bash
export OPENAI_API_BASE="..."
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."

bash run_multi_cwah.sh
bash run_multi_cook.sh
```
Notes:
- `run_multi_cwah.sh` expects `--variant` or `--runs` arguments (repeatable).
- `run_multi_cook.sh` requires at least one `--data-root` pointing to a Cook-MultiPlayer run.

1) Build windows (no LLM calls):
```bash
python llmjudge.py build-windows \
  --input cwah-0-agent-1-human/runs/LLMs_act_Qwen2.5-7B-Instruct_0_task0 \
  --out windows_task0.jsonl \
  --preview-out windows_task0_preview.txt \
  --agent-id 0 --human-id 1
```

2) Build prompts from windows (no LLM calls, two options):
- 2a) full prompts (larger, single file for scoring):
```bash
python llmjudge.py build-prompts \
  --windows windows_task0.jsonl \
  --out prompts_task0.jsonl
```
- 2b) thin requests (smaller; scoring needs `--windows` to refill window_text):
```bash
python llmjudge.py build-prompts \
  --windows windows_task0.jsonl \
  --out requests_task0.jsonl \
  --thin
```

3) Score prompts (OpenAI-compatible API required):
```bash
export OPENAI_API_BASE="http://<host>:<port>/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"

python llmjudge.py score-prompts \
  --prompts prompts_task0.jsonl \
  --out results_task0.jsonl \
  --resume
```

Scoring with thin requests (recommended for storage):
```bash
python llmjudge.py score-prompts \
  --prompts requests_task0.jsonl \
  --windows windows_task0.jsonl \
  --out results_task0_lite.jsonl \
  --resume
```

4) Aggregate to trajectory scores (per dimension):
- Paper default formula: `score = 5 - a*(violation_count / n) - b*(max(5 - window_score)) - msg_penalty`, where `n` is #windows.
- Paper default parameters: `a=0.8`, `b=0.8`, `msg_penalty_threshold=0.15`, `msg_penalty_k=4.0`, `msg_penalty_max=2.0`.
```bash
python llmjudge.py aggregate-results \
  --results results_task0_lite.jsonl \
  --out trajectory_task0.jsonl \
  --out-csv trajectory_task0.csv \
  --a 0.8 --b 0.8
```

5) Export window-level CSV:
```bash
python llmjudge.py export-window-csv \
  --results results_task0_lite.jsonl \
  --out-csv window_scores_task0.csv
```

6) Merge detailed + compact results:
```bash
python llmjudge.py merge-outputs \
  --windows windows_task0.jsonl \
  --results results_task0_lite.jsonl \
  --out-detailed results_task0_detailed.jsonl \
  --out-compact results_task0_compact.jsonl
```

One-shot full pipeline (recommended):
```bash
python llmjudge.py run-pipeline \
  --variant cwah-0-agent-1-human_deepseekduida \
  --agent-id 0 --human-id 1 \
  --api-base "$OPENAI_API_BASE" --model deepseek --api-key "$OPENAI_API_KEY" \
  --workers 4 --resume --a 0.8 --b 0.8
```

Or specify runs directly (if outdir is omitted, it writes to `out_<runs parent>/`):
```bash
python llmjudge.py run-pipeline \
  --runs cwah-0-agent-1-human_deepseekduida/runs \
  --agent-id 0 --human-id 1 \
  --api-base "$OPENAI_API_BASE" --model deepseek --api-key "$OPENAI_API_KEY" \
  --workers 4 --resume --a 0.8 --b 0.8
```

7) Model-level summary (mean over trajectories):
```bash
python llmjudge.py model-summary \
  --trajectory <out_dir>/trajectory.jsonl \
  --out-json <out_dir>/model_summary.json \
  --out-csv <out_dir>/model_summary.csv
```

8) Multi-model comparison (combine out_* into one table):
```bash
python llmjudge.py compare-model-summaries \
  --base Evaluation/evaluation --glob "out_*" \
  --out-csv model_compare.csv \
  --sort-by overall_of_dimension_means --descending
```

## Optional tools (paper/sweep utilities)
Helper modules and optional scripts live under `tools/` and are auto-loaded by the main CLIs.
