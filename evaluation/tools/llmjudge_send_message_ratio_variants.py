#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llmjudge_logio import iter_log_paths, load_log


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _infer_ids_from_variant_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Supported patterns:
      - cwah-<A>-agent-<H>-human_...
      - cwah-<H>-human-<A>-agent_...
    Returns (agent_id, human_id).
    """
    m = re.match(r"^cwah-([0-9]+)-agent-([0-9]+)-human(?:_|$)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^cwah-([0-9]+)-human-([0-9]+)-agent(?:_|$)", name)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None, None


def _count_send_message_in_plans(llm_entries: Any) -> tuple[int, int]:
    if not isinstance(llm_entries, list) or not llm_entries:
        return 0, 0
    total = 0
    msg = 0
    for e in llm_entries:
        if not isinstance(e, dict):
            continue
        total += 1
        plan = str(e.get("plan", "") or "")
        if "[send_message]" in plan:
            msg += 1
    return msg, total


def _count_send_message_in_env_actions(actions: Any) -> tuple[int, int]:
    if not isinstance(actions, list) or not actions:
        return 0, 0
    total = 0
    msg = 0
    for a in actions:
        if not isinstance(a, str):
            continue
        total += 1
        if "[send_message]" in a:
            msg += 1
    return msg, total


def _ratio(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute per-variant send_message ratio (high-level action via LLM `plan`) across all cwah-* variants that contain runs/ logs."
    )
    ap.add_argument("--base", default="coela_11/CoELA", help="Base directory containing cwah-* variant folders.")
    ap.add_argument("--glob", default="cwah-*", help="Glob under --base to search for variants (default: cwah-*).")
    ap.add_argument("--out-csv", default="coela_11/CoELA/evaluation/send_message_ratio_variants.csv")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of log files per variant.")
    args = ap.parse_args(argv)

    base = Path(args.base).expanduser()
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    out_csv = Path(args.out_csv).expanduser()
    if not out_csv.is_absolute():
        out_csv = (Path.cwd() / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    variants = sorted([p for p in base.glob(args.glob) if p.is_dir()])
    if not variants:
        raise SystemExit(f"No variants matched: {base}/{args.glob}")

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for vdir in variants:
        name = vdir.name
        runs_dir = vdir / "runs"
        if not runs_dir.exists():
            continue

        agent_id, human_id = _infer_ids_from_variant_name(name)
        if agent_id is None or human_id is None:
            skipped.append(f"{name}: cannot infer agent_id/human_id from name")
            continue

        log_paths = iter_log_paths(runs_dir, args.max_files)
        if not log_paths:
            skipped.append(f"{name}: no logs_agent_*.pik under runs/")
            continue

        # Per-trajectory ratios (mean of ratios)
        a_llm_ratios: List[float] = []
        h_llm_ratios: List[float] = []
        a_env_ratios: List[float] = []
        h_env_ratios: List[float] = []

        # Global ratios (sum counts / sum totals)
        a_llm_msg_sum = 0
        a_llm_total_sum = 0
        h_llm_msg_sum = 0
        h_llm_total_sum = 0
        a_env_msg_sum = 0
        a_env_total_sum = 0
        h_env_msg_sum = 0
        h_env_total_sum = 0

        task_names: set[str] = set()
        for pik_path in log_paths:
            log = load_log(pik_path)
            task = log.get("task_name")
            if isinstance(task, str) and task:
                task_names.add(task)

            llm = log.get("LLM", {}) if isinstance(log.get("LLM"), dict) else {}
            actions_by_id = log.get("action", {}) if isinstance(log.get("action"), dict) else {}

            a_llm_msg, a_llm_total = _count_send_message_in_plans(llm.get(agent_id, []))
            h_llm_msg, h_llm_total = _count_send_message_in_plans(llm.get(human_id, []))
            a_env_msg, a_env_total = _count_send_message_in_env_actions(actions_by_id.get(agent_id, []))
            h_env_msg, h_env_total = _count_send_message_in_env_actions(actions_by_id.get(human_id, []))

            a_llm_msg_sum += a_llm_msg
            a_llm_total_sum += a_llm_total
            h_llm_msg_sum += h_llm_msg
            h_llm_total_sum += h_llm_total
            a_env_msg_sum += a_env_msg
            a_env_total_sum += a_env_total
            h_env_msg_sum += h_env_msg
            h_env_total_sum += h_env_total

            r = _ratio(a_llm_msg, a_llm_total)
            if isinstance(r, float):
                a_llm_ratios.append(r)
            r = _ratio(h_llm_msg, h_llm_total)
            if isinstance(r, float):
                h_llm_ratios.append(r)
            r = _ratio(a_env_msg, a_env_total)
            if isinstance(r, float):
                a_env_ratios.append(r)
            r = _ratio(h_env_msg, h_env_total)
            if isinstance(r, float):
                h_env_ratios.append(r)

        rows.append(
            {
                "variant": name,
                "runs_dir": str(runs_dir),
                "agent_id": agent_id,
                "human_id": human_id,
                "n_logs": len(log_paths),
                "task_names": ",".join(sorted(task_names)) if task_names else "",
                "agent_llm_send_message_ratio_mean": _mean(a_llm_ratios),
                "human_llm_send_message_ratio_mean": _mean(h_llm_ratios),
                "agent_llm_send_message_ratio_global": _ratio(a_llm_msg_sum, a_llm_total_sum),
                "human_llm_send_message_ratio_global": _ratio(h_llm_msg_sum, h_llm_total_sum),
                "agent_llm_calls_sum": a_llm_total_sum,
                "human_llm_calls_sum": h_llm_total_sum,
                "agent_llm_send_message_count_sum": a_llm_msg_sum,
                "human_llm_send_message_count_sum": h_llm_msg_sum,
                "agent_env_send_message_ratio_mean": _mean(a_env_ratios),
                "human_env_send_message_ratio_mean": _mean(h_env_ratios),
                "agent_env_send_message_ratio_global": _ratio(a_env_msg_sum, a_env_total_sum),
                "human_env_send_message_ratio_global": _ratio(h_env_msg_sum, h_env_total_sum),
                "agent_env_steps_sum": a_env_total_sum,
                "human_env_steps_sum": h_env_total_sum,
                "agent_env_send_message_steps_sum": a_env_msg_sum,
                "human_env_send_message_steps_sum": h_env_msg_sum,
            }
        )

    fieldnames = [
        "variant",
        "runs_dir",
        "agent_id",
        "human_id",
        "n_logs",
        "task_names",
        "agent_llm_send_message_ratio_mean",
        "human_llm_send_message_ratio_mean",
        "agent_llm_send_message_ratio_global",
        "human_llm_send_message_ratio_global",
        "agent_llm_calls_sum",
        "human_llm_calls_sum",
        "agent_llm_send_message_count_sum",
        "human_llm_send_message_count_sum",
        "agent_env_send_message_ratio_mean",
        "human_env_send_message_ratio_mean",
        "agent_env_send_message_ratio_global",
        "human_env_send_message_ratio_global",
        "agent_env_steps_sum",
        "human_env_steps_sum",
        "agent_env_send_message_steps_sum",
        "human_env_send_message_steps_sum",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"wrote_csv={out_csv} rows={len(rows)}")
    if skipped:
        print("skipped:")
        for s in skipped[:50]:
            print("  -", s)
        if len(skipped) > 50:
            print(f"  ... and {len(skipped) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

