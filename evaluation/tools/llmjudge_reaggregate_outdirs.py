#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


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


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


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
        reader = csv.DictReader(f)
        for row in reader:
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


def _ab_for_dim(args: argparse.Namespace, dim: str) -> Tuple[float, float]:
    if dim == "helpfulness":
        a = args.a_helpfulness if args.a_helpfulness is not None else args.a
        b = args.b_helpfulness if args.b_helpfulness is not None else args.b
        return float(a), float(b)
    if dim == "trustfulness":
        a = args.a_trustfulness if args.a_trustfulness is not None else args.a
        b = args.b_trustfulness if args.b_trustfulness is not None else args.b
        return float(a), float(b)
    if dim == "empathy":
        a = args.a_empathy if args.a_empathy is not None else args.a
        b = args.b_empathy if args.b_empathy is not None else args.b
        return float(a), float(b)
    return float(args.a), float(args.b)


def _rewrite_trajectory_fast(
    *,
    traj_jsonl: Path,
    traj_csv: Path,
    args: argparse.Namespace,
    variant_llm_ratio_global: Dict[str, float],
) -> None:
    rows: List[Dict[str, Any]] = []
    with traj_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    with traj_jsonl.open("w", encoding="utf-8") as out_f:
        for tr in rows:
            obj = tr.get("objective")
            ratio = _safe_float(tr.get("agent_send_message_ratio"))
            if ratio is None and isinstance(obj, dict):
                ratio = _safe_float(obj.get("agent_send_message_steps"))
                den = _safe_float(obj.get("agent_env_steps"))
                ratio = (ratio / den) if (ratio is not None and den is not None and den > 0) else None
            if variant_llm_ratio_global:
                variant = _variant_from_log_path(str(tr.get("log_path") or ""))
                if variant and variant in variant_llm_ratio_global:
                    ratio = float(variant_llm_ratio_global[variant])
            pen = _msg_penalty_from_ratio(
                ratio,
                threshold=float(args.msg_penalty_threshold),
                k=float(args.msg_penalty_k),
                max_penalty=float(args.msg_penalty_max),
            )
            tr["agent_send_message_ratio"] = ratio
            tr["agent_send_message_ratio_env"] = ratio
            tr["agent_send_message_ratio_llm_global"] = ratio
            tr["msg_penalty"] = pen
            tr["weights"] = {
                "a": args.a,
                "b": args.b,
                "helpfulness": {"a": args.a_helpfulness, "b": args.b_helpfulness},
                "trustfulness": {"a": args.a_trustfulness, "b": args.b_trustfulness},
                "empathy": {"a": args.a_empathy, "b": args.b_empathy},
            }

            dims = tr.get("dimensions")
            if isinstance(dims, dict):
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
                    a, b = _ab_for_dim(args, dim)
                    base = 5.0 - float(a) * (float(viol or 0.0) / float(n)) - float(b) * float(gap or 0.0)
                    base = base - float(pen or 0.0)
                    drec["score"] = max(0.0, min(5.0, base))

            out_f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    # Keep trajectory.csv consistent with current llmjudge_aggregate_results.py output.
    fieldnames = [
        "log_path",
        "agent_id",
        "human_id",
        "task_id",
        "env_id",
        "task_name",
        "n_windows",
        "helpfulness_score",
        "trustfulness_score",
        "empathy_score",
        "helpfulness_violation_count_sum",
        "trustfulness_violation_count_sum",
        "empathy_violation_count_sum",
        "helpfulness_max_window_score_gap",
        "trustfulness_max_window_score_gap",
        "empathy_max_window_score_gap",
        "agent_send_message_ratio",
        "agent_send_message_ratio_env",
        "agent_send_message_ratio_env_steps",
        "agent_send_message_ratio_llm",
        "agent_send_message_ratio_llm_global",
        "msg_penalty",
        "model_output_tokens_mean",
        "model_output_tokens_std",
        "model_output_tokens_sum",
        "model_output_tokens_n_calls",
        "model_output_tokenizer",
    ]
    traj_csv.parent.mkdir(parents=True, exist_ok=True)
    with traj_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for tr in rows:
            dims = tr.get("dimensions") if isinstance(tr.get("dimensions"), dict) else {}
            mt = tr.get("model_output_tokens") if isinstance(tr.get("model_output_tokens"), dict) else {}
            w.writerow(
                {
                    "log_path": tr.get("log_path"),
                    "agent_id": _safe_int(tr.get("agent_id"), -1),
                    "human_id": _safe_int(tr.get("human_id"), -1),
                    "task_id": tr.get("task_id"),
                    "env_id": tr.get("env_id"),
                    "task_name": tr.get("task_name"),
                    "n_windows": tr.get("n_windows"),
                    "helpfulness_score": (dims.get("helpfulness") or {}).get("score") if isinstance(dims.get("helpfulness"), dict) else None,
                    "trustfulness_score": (dims.get("trustfulness") or {}).get("score") if isinstance(dims.get("trustfulness"), dict) else None,
                    "empathy_score": (dims.get("empathy") or {}).get("score") if isinstance(dims.get("empathy"), dict) else None,
                    "helpfulness_violation_count_sum": (dims.get("helpfulness") or {}).get("violation_count_sum") if isinstance(dims.get("helpfulness"), dict) else None,
                    "trustfulness_violation_count_sum": (dims.get("trustfulness") or {}).get("violation_count_sum") if isinstance(dims.get("trustfulness"), dict) else None,
                    "empathy_violation_count_sum": (dims.get("empathy") or {}).get("violation_count_sum") if isinstance(dims.get("empathy"), dict) else None,
                    "helpfulness_max_window_score_gap": (dims.get("helpfulness") or {}).get("max_window_score_gap") if isinstance(dims.get("helpfulness"), dict) else None,
                    "trustfulness_max_window_score_gap": (dims.get("trustfulness") or {}).get("max_window_score_gap") if isinstance(dims.get("trustfulness"), dict) else None,
                    "empathy_max_window_score_gap": (dims.get("empathy") or {}).get("max_window_score_gap") if isinstance(dims.get("empathy"), dict) else None,
                    "agent_send_message_ratio": tr.get("agent_send_message_ratio"),
                    "agent_send_message_ratio_env": tr.get("agent_send_message_ratio_env"),
                    "agent_send_message_ratio_env_steps": tr.get("agent_send_message_ratio_env_steps"),
                    "agent_send_message_ratio_llm": tr.get("agent_send_message_ratio_llm"),
                    "agent_send_message_ratio_llm_global": tr.get("agent_send_message_ratio_llm_global"),
                    "msg_penalty": tr.get("msg_penalty"),
                    "model_output_tokens_mean": mt.get("tokens_mean"),
                    "model_output_tokens_std": mt.get("tokens_std"),
                    "model_output_tokens_sum": mt.get("tokens_sum"),
                    "model_output_tokens_n_calls": mt.get("n_calls_with_text"),
                    "model_output_tokenizer": mt.get("tokenizer"),
                }
            )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-aggregate trajectory + model_summary from existing results_lite_obj.jsonl (no judge API calls)."
    )
    ap.add_argument("--base", required=True, help="Directory containing many out_* folders.")
    ap.add_argument("--glob", default="out_*", help='Glob pattern under --base (default: "out_*").')
    ap.add_argument("--a", type=float, default=1.0, help="Default a (viol_count/n) coefficient.")
    ap.add_argument("--b", type=float, default=1.0, help="Default b (worst gap) coefficient.")
    ap.add_argument("--a-helpfulness", type=float, default=None)
    ap.add_argument("--b-helpfulness", type=float, default=None)
    ap.add_argument("--a-trustfulness", type=float, default=None)
    ap.add_argument("--b-trustfulness", type=float, default=None)
    ap.add_argument("--a-empathy", type=float, default=None)
    ap.add_argument("--b-empathy", type=float, default=None)
    ap.add_argument("--msg-penalty-threshold", type=float, default=0.15)
    ap.add_argument("--msg-penalty-max", type=float, default=1.0)
    ap.add_argument("--msg-penalty-k", type=float, default=4.0)
    ap.add_argument(
        "--global-send-message-csv",
        default=None,
        help="If set, overrides agent_send_message_ratio with agent_llm_send_message_ratio_global from this CSV (by variant extracted from log_path).",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Fast path: recompute only final dimension scores from existing trajectory.jsonl (keeps cached token stats).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    base = Path(args.base).expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"--base not found: {base}")

    script_dir = Path(__file__).resolve().parent
    llmjudge = script_dir.parent / "llmjudge.py"

    variant_llm_ratio_global = _load_variant_global_llm_send_message_ratio(args.global_send_message_csv)

    outdirs = sorted([p for p in base.glob(args.glob) if p.is_dir()])
    if not outdirs:
        raise SystemExit(f"No outdirs matched: {base}/{args.glob}")

    for outdir in outdirs:
        results = outdir / "results_lite_obj.jsonl"
        if not results.exists():
            continue
        traj = outdir / "trajectory.jsonl"
        traj_csv = outdir / "trajectory.csv"
        msj = outdir / "model_summary.json"
        msc = outdir / "model_summary.csv"

        if args.dry_run:
            print("[dry-run]", outdir.name, "(fast)" if args.fast else "(full)")
        else:
            if args.fast and traj.exists():
                _rewrite_trajectory_fast(
                    traj_jsonl=traj,
                    traj_csv=traj_csv,
                    args=args,
                    variant_llm_ratio_global=variant_llm_ratio_global,
                )
            else:
                cmd = [
                    "python",
                    str(llmjudge),
                    "aggregate-results",
                    "--results",
                    str(results),
                    "--out",
                    str(traj),
                    "--out-csv",
                    str(traj_csv),
                    "--a",
                    str(args.a),
                    "--b",
                    str(args.b),
                    "--msg-penalty-threshold",
                    str(args.msg_penalty_threshold),
                    "--msg-penalty-max",
                    str(args.msg_penalty_max),
                    "--msg-penalty-k",
                    str(args.msg_penalty_k),
                ]
                if args.global_send_message_csv:
                    cmd += ["--global-send-message-csv", str(args.global_send_message_csv)]
                for flag, val in [
                    ("--a-helpfulness", args.a_helpfulness),
                    ("--b-helpfulness", args.b_helpfulness),
                    ("--a-trustfulness", args.a_trustfulness),
                    ("--b-trustfulness", args.b_trustfulness),
                    ("--a-empathy", args.a_empathy),
                    ("--b-empathy", args.b_empathy),
                ]:
                    if val is not None:
                        cmd += [flag, str(val)]
                _run(cmd)
            _run(
                [
                    "python",
                    str(llmjudge),
                    "model-summary",
                    "--trajectory",
                    str(traj),
                    "--out-json",
                    str(msj),
                    "--out-csv",
                    str(msc),
                ]
            )
            print(f"[ok] {outdir.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
