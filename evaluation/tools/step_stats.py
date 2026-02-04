#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_log_paths(input_path: Path, max_files: Optional[int]) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    paths = sorted(input_path.rglob("logs_agent_*.pik"))
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def _load(p: Path) -> Dict[str, Any]:
    return pickle.loads(p.read_bytes())


def _count_action(actions: Any, token: str) -> int:
    if not isinstance(actions, list):
        return 0
    return sum(1 for a in actions if isinstance(a, str) and token in a)


def _objective_steps(log: Dict[str, Any], agent_id: int, human_id: int) -> Dict[str, Any]:
    actions_by_id = log.get("action", {})
    agent_actions = actions_by_id.get(agent_id) if isinstance(actions_by_id, dict) else None
    human_actions = actions_by_id.get(human_id) if isinstance(actions_by_id, dict) else None

    agent_env_steps = len(agent_actions) if isinstance(agent_actions, list) else None
    human_env_steps = len(human_actions) if isinstance(human_actions, list) else None

    total_env_steps = None
    if agent_env_steps is not None:
        total_env_steps = agent_env_steps
    elif isinstance(log.get("progress"), list):
        total_env_steps = len(log["progress"])
    elif isinstance(log.get("goals_finished"), list):
        total_env_steps = len(log["goals_finished"])

    llm = log.get("LLM", {})
    agent_llm_calls = len(llm.get(agent_id, [])) if isinstance(llm, dict) else None
    human_llm_calls = len(llm.get(human_id, [])) if isinstance(llm, dict) else None

    finished = log.get("finished")
    # LLM-level message selection rate (how often the model decided to send a message).
    llm = log.get("LLM", {})
    agent_llm_entries = llm.get(agent_id, []) if isinstance(llm, dict) else []
    human_llm_entries = llm.get(human_id, []) if isinstance(llm, dict) else []

    def _llm_message_rate(entries: Any) -> Tuple[int, int, Optional[float]]:
        if not isinstance(entries, list) or not entries:
            return 0, 0, None
        total = 0
        msg = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            total += 1
            plan = str(e.get("plan", "") or "")
            if "[send_message]" in plan:
                msg += 1
        return msg, total, (msg / total if total > 0 else None)

    agent_msg_llm, agent_total_llm, agent_msg_rate = _llm_message_rate(agent_llm_entries)
    human_msg_llm, human_total_llm, human_msg_rate = _llm_message_rate(human_llm_entries)

    return {
        "total_env_steps": total_env_steps,
        "agent_env_steps": agent_env_steps,
        "human_env_steps": human_env_steps,
        "agent_send_message_steps": _count_action(agent_actions, "[send_message]"),
        "human_send_message_steps": _count_action(human_actions, "[send_message]"),
        "agent_llm_calls": agent_llm_calls,
        "human_llm_calls": human_llm_calls,
        "agent_llm_message_count": agent_msg_llm,
        "agent_llm_message_rate": agent_msg_rate,
        "human_llm_message_count": human_msg_llm,
        "human_llm_message_rate": human_msg_rate,
        "finished": finished if isinstance(finished, bool) else None,
    }


def _mean(nums: List[float]) -> Optional[float]:
    if not nums:
        return None
    return sum(nums) / len(nums)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute objective step statistics for a variant runs directory.")
    ap.add_argument("--variant", required=True, help="Variant dir (contains runs/) or runs dir itself.")
    ap.add_argument("--agent-id", type=int, default=0)
    ap.add_argument("--human-id", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--out-csv", default=None, help="Optional per-trajectory CSV output.")

    args = ap.parse_args(argv)
    variant = Path(args.variant).expanduser().resolve()
    runs_dir = variant / "runs" if (variant / "runs").exists() else variant

    paths = _iter_log_paths(runs_dir, args.max_files)
    if not paths:
        raise SystemExit(f"No logs_agent_*.pik under: {runs_dir}")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        log = _load(p)
        obj = _objective_steps(log, args.agent_id, args.human_id)
        rows.append(
            {
                "log_path": str(p),
                "task_id": log.get("task_id"),
                "env_id": log.get("env_id"),
                "task_name": log.get("task_name"),
                "agent_id": args.agent_id,
                "human_id": args.human_id,
                **obj,
            }
        )

    steps = [r["total_env_steps"] for r in rows if isinstance(r.get("total_env_steps"), int)]
    finished = [r["finished"] for r in rows if isinstance(r.get("finished"), bool)]
    agent_msg_rates = [r["agent_llm_message_rate"] for r in rows if isinstance(r.get("agent_llm_message_rate"), float)]
    human_msg_rates = [r["human_llm_message_rate"] for r in rows if isinstance(r.get("human_llm_message_rate"), float)]

    print(f"files={len(rows)}")
    print(f"mean_total_env_steps={_mean([float(x) for x in steps])}")
    if finished:
        print(f"finished_rate={sum(1 for x in finished if x)/len(finished)}")
    if agent_msg_rates:
        print(f"agent_mean_llm_message_rate={_mean(agent_msg_rates)}")
    if human_msg_rates:
        print(f"human_mean_llm_message_rate={_mean(human_msg_rates)}")

    if args.out_csv:
        out = Path(args.out_csv).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "log_path",
                    "task_id",
                    "env_id",
                    "task_name",
                    "agent_id",
                    "human_id",
                    "total_env_steps",
                    "agent_env_steps",
                    "human_env_steps",
                    "agent_send_message_steps",
                    "human_send_message_steps",
                    "agent_llm_calls",
                    "human_llm_calls",
                    "agent_llm_message_count",
                    "agent_llm_message_rate",
                    "human_llm_message_count",
                    "human_llm_message_rate",
                    "finished",
                ],
            )
            w.writeheader()
            w.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
