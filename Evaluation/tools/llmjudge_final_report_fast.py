#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _std_pop(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(v)


def _variant_from_log_path(log_path: str) -> Optional[str]:
    if not log_path:
        return None
    s = str(log_path)
    m = re.search(r"/(cwah-[^/]+)/runs/", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(cwah-[^/]+)\b", s)
    return m.group(1) if m else None


def _load_variant_global_llm_send_message_ratio(csv_path: Optional[str]) -> Dict[str, float]:
    if not csv_path:
        return {}
    p = Path(os.path.expanduser(str(csv_path))).resolve()
    if not p.exists():
        return {}
    out: Dict[str, float] = {}
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            variant = (row.get("variant") or "").strip()
            val = _safe_float(row.get("agent_llm_send_message_ratio_global"))
            if variant and val is not None:
                out[variant] = float(val)
    return out


def _msg_penalty_from_ratio(
    ratio: Optional[float],
    *,
    threshold: float,
    k: float,
    max_penalty: float,
) -> float:
    if ratio is None:
        return 0.0
    if threshold <= 0 or max_penalty <= 0:
        return 0.0

    r = max(0.0, min(1.0, float(ratio)))
    if r >= threshold:
        return 0.0

    x = (threshold - r) / threshold  # in (0, 1]
    kk = float(k)
    if kk <= 0:
        return max(0.0, min(max_penalty, max_penalty * x))

    denom = math.exp(kk) - 1.0
    if denom <= 0:
        return 0.0
    val = (math.exp(kk * x) - 1.0) / denom
    return max(0.0, min(max_penalty, max_penalty * val))


def _recompute_scores_for_trajectory(
    tr: Dict[str, Any],
    *,
    a: float,
    b: float,
    msg_penalty: float,
) -> Tuple[Dict[str, Any], Optional[float]]:
    dims = tr.get("dimensions")
    if not isinstance(dims, dict):
        return tr, None

    per_dim_scores: List[float] = []
    for dim in ["helpfulness", "trustfulness", "empathy"]:
        drec = dims.get(dim)
        if not isinstance(drec, dict):
            continue
        n = _safe_float(drec.get("n"))
        viol = _safe_float(drec.get("violation_count_sum"))
        gap = _safe_float(drec.get("max_window_score_gap"))
        if n is None or n <= 0:
            drec["score"] = None
            continue
        base = 5.0 - float(a) * (float(viol or 0.0) / float(n)) - float(b) * float(gap or 0.0)
        base = base - float(msg_penalty or 0.0)
        score = max(0.0, min(5.0, base))
        drec["score"] = score
        per_dim_scores.append(score)

    overall = _mean(per_dim_scores)
    return tr, overall


def _find_outdirs(base: Path, pattern: str) -> List[Path]:
    return sorted([p for p in base.glob(pattern) if p.is_dir()])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fast final report: recompute scores from cached trajectory.jsonl only (no judge, no logs)."
    )
    ap.add_argument(
        "--bases",
        nargs="+",
        default=["coela_11/CoELA/evaluation", "coela_11/CoELA/evaluation/proagent"],
        help="One or more base dirs containing out_* folders.",
    )
    ap.add_argument("--glob", default="out_*", help='Outdir glob under each base (default: "out_*").')
    ap.add_argument("--out-csv", default="coela_11/CoELA/evaluation/final_report_all.csv")
    ap.add_argument("--out-task-csv", default="coela_11/CoELA/evaluation/final_report_all_tasks.csv")
    ap.add_argument("--include-outdir", action="store_true", help="Include absolute outdir path in CSVs.")

    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--b", type=float, required=True)
    ap.add_argument("--msg-penalty-threshold", type=float, default=0.2)
    ap.add_argument("--msg-penalty-k", type=float, default=4.0)
    ap.add_argument("--msg-penalty-max", type=float, default=1.0, help="This is your tunable p.")
    ap.add_argument(
        "--global-send-message-csv",
        default="coela_11/CoELA/evaluation/send_message_ratio_all_variants.csv",
        help="Uses agent_llm_send_message_ratio_global as the penalty ratio (when variant is matched).",
    )
    args = ap.parse_args(argv)

    variant_llm_ratio_global = _load_variant_global_llm_send_message_ratio(args.global_send_message_csv)

    rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    for base_s in args.bases:
        base = Path(base_s).expanduser().resolve()
        outdirs = _find_outdirs(base, args.glob)
        for outdir in outdirs:
            traj_path = outdir / "trajectory.jsonl"
            if not traj_path.exists():
                continue

            trajs = list(_iter_jsonl(traj_path))

            # Compute penalty ratio per-trajectory (prefer variant-level global LLM ratio if possible).
            overall_scores: List[float] = []
            dim_scores: Dict[str, List[float]] = {"helpfulness": [], "trustfulness": [], "empathy": []}
            penalties: List[float] = []
            ratios_used: List[float] = []

            for tr in trajs:
                log_path = str(tr.get("log_path") or "")
                ratio = None
                if variant_llm_ratio_global:
                    variant = _variant_from_log_path(log_path)
                    if variant and variant in variant_llm_ratio_global:
                        ratio = float(variant_llm_ratio_global[variant])
                if ratio is None:
                    ratio = _safe_float(tr.get("agent_send_message_ratio_env"))
                if ratio is None:
                    ratio = _safe_float(tr.get("agent_send_message_ratio"))

                pen = _msg_penalty_from_ratio(
                    ratio,
                    threshold=float(args.msg_penalty_threshold),
                    k=float(args.msg_penalty_k),
                    max_penalty=float(args.msg_penalty_max),
                )
                penalties.append(pen)
                if ratio is not None:
                    ratios_used.append(float(ratio))

                tr, overall = _recompute_scores_for_trajectory(tr, a=float(args.a), b=float(args.b), msg_penalty=pen)
                if overall is not None:
                    overall_scores.append(overall)
                dims = tr.get("dimensions") if isinstance(tr.get("dimensions"), dict) else {}
                for d in ["helpfulness", "trustfulness", "empathy"]:
                    s = _safe_float((dims.get(d) or {}).get("score") if isinstance(dims.get(d), dict) else None)
                    if s is not None:
                        dim_scores[d].append(s)

            helpful = _mean(dim_scores["helpfulness"])
            trust = _mean(dim_scores["trustfulness"])
            empath = _mean(dim_scores["empathy"])
            overall_of_means = _mean([x for x in [helpful, trust, empath] if x is not None])

            base_str = str(base)
            row: Dict[str, Any] = {
                "base": base_str,
                "name": outdir.name,
                "n_trajectories": len(trajs),
                "helpfulness_mean": helpful,
                "trustfulness_mean": trust,
                "empathy_mean": empath,
                "overall_of_dimension_means": overall_of_means,
                "overall_per_trajectory_mean": _mean(overall_scores),
                "overall_per_trajectory_std": _std_pop(overall_scores),
                "agent_send_message_ratio_mean": _mean(ratios_used),
                "agent_send_message_ratio_n": len(ratios_used),
                "msg_penalty_mean": _mean(penalties),
                "a": float(args.a),
                "b": float(args.b),
                "msg_penalty_threshold": float(args.msg_penalty_threshold),
                "msg_penalty_k": float(args.msg_penalty_k),
                "msg_penalty_max": float(args.msg_penalty_max),
            }
            if args.include_outdir:
                row["outdir"] = str(outdir)
            rows.append(row)

            # Task-level aggregates
            by_task: Dict[str, List[Dict[str, Any]]] = {}
            for tr in trajs:
                task = tr.get("task_name")
                if isinstance(task, str) and task:
                    by_task.setdefault(task, []).append(tr)
            for task, t_trajs in sorted(by_task.items(), key=lambda kv: kv[0]):
                overall_scores_t: List[float] = []
                dim_scores_t: Dict[str, List[float]] = {"helpfulness": [], "trustfulness": [], "empathy": []}
                ratios_t: List[float] = []
                pens_t: List[float] = []
                for tr in t_trajs:
                    ratio = _safe_float(tr.get("agent_send_message_ratio_env"))
                    if ratio is not None:
                        ratios_t.append(float(ratio))
                    pen = _safe_float(tr.get("msg_penalty"))
                    if pen is not None:
                        pens_t.append(float(pen))
                    dims = tr.get("dimensions") if isinstance(tr.get("dimensions"), dict) else {}
                    per_dim: List[float] = []
                    for d in ["helpfulness", "trustfulness", "empathy"]:
                        s = _safe_float((dims.get(d) or {}).get("score") if isinstance(dims.get(d), dict) else None)
                        if s is not None:
                            dim_scores_t[d].append(s)
                            per_dim.append(s)
                    o = _mean(per_dim)
                    if o is not None:
                        overall_scores_t.append(o)

                trow: Dict[str, Any] = {
                    "base": base_str,
                    "name": outdir.name,
                    "task_name": task,
                    "n_trajectories": len(t_trajs),
                    "overall_mean": _mean(overall_scores_t),
                    "helpfulness_mean": _mean(dim_scores_t["helpfulness"]),
                    "trustfulness_mean": _mean(dim_scores_t["trustfulness"]),
                    "empathy_mean": _mean(dim_scores_t["empathy"]),
                    "agent_send_message_ratio_mean": _mean(ratios_t),
                    "msg_penalty_mean": _mean(pens_t),
                }
                if args.include_outdir:
                    trow["outdir"] = str(outdir)
                task_rows.append(trow)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_task_csv = Path(args.out_task_csv).expanduser().resolve()
    out_task_csv.parent.mkdir(parents=True, exist_ok=True)
    task_fieldnames = sorted({k for r in task_rows for k in r.keys()}) if task_rows else []
    with out_task_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=task_fieldnames)
        w.writeheader()
        for r in task_rows:
            w.writerow(r)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_task_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

