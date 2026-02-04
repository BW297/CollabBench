#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmjudge_logio import iter_log_paths, load_log


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _count_send_message_in_plans(llm_entries: Any) -> tuple[int, int, Optional[float]]:
    """
    Returns (send_message_count, total_llm_calls, ratio) based on LLM entries' `plan` field.
    This is the "high-level action" send_message selection rate.
    """
    if not isinstance(llm_entries, list) or not llm_entries:
        return 0, 0, None
    total = 0
    msg = 0
    for e in llm_entries:
        if not isinstance(e, dict):
            continue
        total += 1
        plan = str(e.get("plan", "") or "")
        if "[send_message]" in plan:
            msg += 1
    return msg, total, (msg / total if total > 0 else None)


def _count_send_message_in_env_actions(actions: Any) -> tuple[int, int, Optional[float]]:
    """
    Returns (send_message_steps, total_env_steps, ratio) based on env `action` strings.
    This is not "high-level action", but is useful for sanity-checking.
    """
    if not isinstance(actions, list) or not actions:
        return 0, 0, None
    total = 0
    msg = 0
    for a in actions:
        if not isinstance(a, str):
            continue
        total += 1
        if "[send_message]" in a:
            msg += 1
    return msg, total, (msg / total if total > 0 else None)


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute send_message selection ratio as a high-level action (from LLM `plan` field) across logs."
    )
    ap.add_argument("--input", required=True, help="Run dir, runs root, or a single `logs_agent_*.pik` file.")
    ap.add_argument("--agent-id", type=int, default=0, help="Evaluated agent id.")
    ap.add_argument("--human-id", type=int, default=1, help="Partner id.")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of log files processed.")
    ap.add_argument("--out-csv", default=None, help="Optional per-trajectory CSV output path.")
    args = ap.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    log_paths = iter_log_paths(input_path, args.max_files)
    if not log_paths:
        raise SystemExit(f"No `logs_agent_*.pik` found under: {input_path}")

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else None
    csv_file = None
    writer = None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = out_csv.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "log_path",
                "task_id",
                "env_id",
                "task_name",
                "agent_id",
                "human_id",
                "agent_llm_calls",
                "agent_llm_send_message_count",
                "agent_llm_send_message_ratio",
                "human_llm_calls",
                "human_llm_send_message_count",
                "human_llm_send_message_ratio",
                "agent_env_steps",
                "agent_env_send_message_steps",
                "agent_env_send_message_ratio",
                "human_env_steps",
                "human_env_send_message_steps",
                "human_env_send_message_ratio",
            ],
        )
        writer.writeheader()

    agent_ratios: List[float] = []
    human_ratios: List[float] = []
    agent_env_ratios: List[float] = []
    human_env_ratios: List[float] = []

    for pik_path in log_paths:
        log = load_log(pik_path)
        llm = log.get("LLM", {}) if isinstance(log.get("LLM"), dict) else {}
        actions_by_id = log.get("action", {}) if isinstance(log.get("action"), dict) else {}

        agent_llm = llm.get(args.agent_id, [])
        human_llm = llm.get(args.human_id, [])
        agent_actions = actions_by_id.get(args.agent_id, [])
        human_actions = actions_by_id.get(args.human_id, [])

        a_msg, a_total, a_ratio = _count_send_message_in_plans(agent_llm)
        h_msg, h_total, h_ratio = _count_send_message_in_plans(human_llm)
        a_env_msg, a_env_total, a_env_ratio = _count_send_message_in_env_actions(agent_actions)
        h_env_msg, h_env_total, h_env_ratio = _count_send_message_in_env_actions(human_actions)

        if isinstance(a_ratio, float):
            agent_ratios.append(a_ratio)
        if isinstance(h_ratio, float):
            human_ratios.append(h_ratio)
        if isinstance(a_env_ratio, float):
            agent_env_ratios.append(a_env_ratio)
        if isinstance(h_env_ratio, float):
            human_env_ratios.append(h_env_ratio)

        if writer:
            writer.writerow(
                {
                    "log_path": str(pik_path),
                    "task_id": _safe_int(log.get("task_id"), -1),
                    "env_id": _safe_int(log.get("env_id"), -1),
                    "task_name": str(log.get("task_name", "")),
                    "agent_id": args.agent_id,
                    "human_id": args.human_id,
                    "agent_llm_calls": a_total,
                    "agent_llm_send_message_count": a_msg,
                    "agent_llm_send_message_ratio": a_ratio,
                    "human_llm_calls": h_total,
                    "human_llm_send_message_count": h_msg,
                    "human_llm_send_message_ratio": h_ratio,
                    "agent_env_steps": a_env_total,
                    "agent_env_send_message_steps": a_env_msg,
                    "agent_env_send_message_ratio": a_env_ratio,
                    "human_env_steps": h_env_total,
                    "human_env_send_message_steps": h_env_msg,
                    "human_env_send_message_ratio": h_env_ratio,
                }
            )

    if csv_file:
        csv_file.close()

    print(f"files={len(log_paths)}")
    print(f"agent_llm_send_message_ratio_mean={_mean(agent_ratios)} (n={len(agent_ratios)})")
    print(f"human_llm_send_message_ratio_mean={_mean(human_ratios)} (n={len(human_ratios)})")
    print(f"agent_env_send_message_ratio_mean={_mean(agent_env_ratios)} (n={len(agent_env_ratios)})")
    print(f"human_env_send_message_ratio_mean={_mean(human_env_ratios)} (n={len(human_env_ratios)})")
    if out_csv:
        print(f"wrote_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

