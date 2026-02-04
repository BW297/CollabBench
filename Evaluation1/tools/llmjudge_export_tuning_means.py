#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_outdirs(base: Path, pattern: str) -> List[Path]:
    # Only scan one level to avoid pulling in unrelated artifacts.
    return sorted([p for p in base.glob(pattern) if p.is_dir()])


def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _as_dict(d: Any) -> Dict[str, Any]:
    return d if isinstance(d, dict) else {}


def _weights_from_first(trajs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not trajs:
        return {}
    w = trajs[0].get("weights")
    return w if isinstance(w, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export per-outdir trajectory-level mean intermediate stats for tuning (violations/n/gaps/msg penalty/tokens)."
    )
    ap.add_argument("--base", default="coela_11/CoELA/evaluation", help="Directory containing out_* folders.")
    ap.add_argument("--glob", default="out_*", help="Glob pattern under --base (default: out_*).")
    ap.add_argument("--out-csv", default="coela_11/CoELA/evaluation/tuning_means.csv", help="Output CSV path.")
    args = ap.parse_args(argv)

    base = _resolve_path(args.base)
    out_csv = _resolve_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    outdirs = _find_outdirs(base, args.glob)
    if not outdirs:
        raise SystemExit(f"No outdirs matched: {base}/{args.glob}")

    fieldnames = [
        "name",
        "outdir",
        "n_trajectories",
        # current scores (means over trajectories)
        "helpfulness_score_mean",
        "trustfulness_score_mean",
        "empathy_score_mean",
        "overall_score_mean",
        # intermediate quantities (means over trajectories)
        "helpfulness_violation_count_sum_mean",
        "trustfulness_violation_count_sum_mean",
        "empathy_violation_count_sum_mean",
        "helpfulness_n_windows_mean",
        "trustfulness_n_windows_mean",
        "empathy_n_windows_mean",
        "helpfulness_max_window_score_gap_mean",
        "trustfulness_max_window_score_gap_mean",
        "empathy_max_window_score_gap_mean",
        "agent_send_message_ratio_env_mean",
        "agent_send_message_ratio_llm_mean",
        "msg_penalty_mean",
        # model output tokens (per LLM call)
        "model_output_tokens_mean_mean",
        "model_output_tokens_std_mean",
        "model_output_tokens_n_calls_mean",
        # objective stats (if present)
        "agent_env_steps_mean",
        "total_env_steps_mean",
        "agent_llm_calls_mean",
        # record weights found in trajectory (first record)
        "a",
        "b",
        "a_helpfulness",
        "b_helpfulness",
        "a_trustfulness",
        "b_trustfulness",
        "a_empathy",
        "b_empathy",
    ]

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for outdir in outdirs:
        traj_path = outdir / "trajectory.jsonl"
        if not traj_path.exists():
            continue
        try:
            trajs = list(_iter_jsonl(traj_path))
        except Exception as e:
            errors.append(f"{outdir}: failed to read trajectory.jsonl: {e}")
            continue

        dims = ["helpfulness", "trustfulness", "empathy"]

        # score means
        score_vals: Dict[str, List[float]] = {d: [] for d in dims}
        overall_scores: List[float] = []

        # intermediate means
        viol_vals: Dict[str, List[float]] = {d: [] for d in dims}
        n_vals: Dict[str, List[float]] = {d: [] for d in dims}
        gap_vals: Dict[str, List[float]] = {d: [] for d in dims}

        msg_ratio_env_vals: List[float] = []
        msg_ratio_llm_vals: List[float] = []
        msg_pen_vals: List[float] = []

        tok_mean_vals: List[float] = []
        tok_std_vals: List[float] = []
        tok_n_calls_vals: List[float] = []

        agent_env_steps_vals: List[float] = []
        total_env_steps_vals: List[float] = []
        agent_llm_calls_vals: List[float] = []

        for tr in trajs:
            drec = _as_dict(tr.get("dimensions"))
            per_dim_scores: List[float] = []
            for d in dims:
                dd = _as_dict(drec.get(d))
                s = _safe_float(dd.get("score"))
                if s is not None:
                    score_vals[d].append(s)
                    per_dim_scores.append(s)
                v = _safe_float(dd.get("violation_count_sum"))
                if v is not None:
                    viol_vals[d].append(v)
                n = _safe_float(dd.get("n"))
                if n is not None:
                    n_vals[d].append(n)
                g = _safe_float(dd.get("max_window_score_gap"))
                if g is not None:
                    gap_vals[d].append(g)

            if per_dim_scores:
                overall_scores.append(sum(per_dim_scores) / len(per_dim_scores))

            mr = _safe_float(tr.get("agent_send_message_ratio"))
            if mr is not None:
                msg_ratio_env_vals.append(mr)
            mr = _safe_float(tr.get("agent_send_message_ratio_env"))
            if mr is not None:
                msg_ratio_env_vals.append(mr)
            mr = _safe_float(tr.get("agent_send_message_ratio_llm"))
            if mr is not None:
                msg_ratio_llm_vals.append(mr)
            mp = _safe_float(tr.get("msg_penalty"))
            if mp is not None:
                msg_pen_vals.append(mp)

            mot = _as_dict(tr.get("model_output_tokens"))
            tm = _safe_float(mot.get("tokens_mean"))
            if tm is not None:
                tok_mean_vals.append(tm)
            ts = _safe_float(mot.get("tokens_std"))
            if ts is not None:
                tok_std_vals.append(ts)
            tn = _safe_float(mot.get("n_calls_with_text"))
            if tn is not None:
                tok_n_calls_vals.append(tn)

            obj = _as_dict(tr.get("objective"))
            v = _safe_float(obj.get("agent_env_steps"))
            if v is not None:
                agent_env_steps_vals.append(v)
            v = _safe_float(obj.get("total_env_steps"))
            if v is not None:
                total_env_steps_vals.append(v)
            v = _safe_float(obj.get("agent_llm_calls"))
            if v is not None:
                agent_llm_calls_vals.append(v)

        w = _weights_from_first(trajs)
        row: Dict[str, Any] = {
            "name": outdir.name,
            "outdir": str(outdir),
            "n_trajectories": len(trajs),
            "helpfulness_score_mean": _mean(score_vals["helpfulness"]),
            "trustfulness_score_mean": _mean(score_vals["trustfulness"]),
            "empathy_score_mean": _mean(score_vals["empathy"]),
            "overall_score_mean": _mean(overall_scores),
            "helpfulness_violation_count_sum_mean": _mean(viol_vals["helpfulness"]),
            "trustfulness_violation_count_sum_mean": _mean(viol_vals["trustfulness"]),
            "empathy_violation_count_sum_mean": _mean(viol_vals["empathy"]),
            "helpfulness_n_windows_mean": _mean(n_vals["helpfulness"]),
            "trustfulness_n_windows_mean": _mean(n_vals["trustfulness"]),
            "empathy_n_windows_mean": _mean(n_vals["empathy"]),
            "helpfulness_max_window_score_gap_mean": _mean(gap_vals["helpfulness"]),
            "trustfulness_max_window_score_gap_mean": _mean(gap_vals["trustfulness"]),
            "empathy_max_window_score_gap_mean": _mean(gap_vals["empathy"]),
            "agent_send_message_ratio_env_mean": _mean(msg_ratio_env_vals),
            "agent_send_message_ratio_llm_mean": _mean(msg_ratio_llm_vals),
            "msg_penalty_mean": _mean(msg_pen_vals),
            "model_output_tokens_mean_mean": _mean(tok_mean_vals),
            "model_output_tokens_std_mean": _mean(tok_std_vals),
            "model_output_tokens_n_calls_mean": _mean(tok_n_calls_vals),
            "agent_env_steps_mean": _mean(agent_env_steps_vals),
            "total_env_steps_mean": _mean(total_env_steps_vals),
            "agent_llm_calls_mean": _mean(agent_llm_calls_vals),
            "a": _as_dict(w).get("a"),
            "b": _as_dict(w).get("b"),
            "a_helpfulness": _as_dict(_as_dict(w).get("helpfulness")).get("a"),
            "b_helpfulness": _as_dict(_as_dict(w).get("helpfulness")).get("b"),
            "a_trustfulness": _as_dict(_as_dict(w).get("trustfulness")).get("a"),
            "b_trustfulness": _as_dict(_as_dict(w).get("trustfulness")).get("b"),
            "a_empathy": _as_dict(_as_dict(w).get("empathy")).get("a"),
            "b_empathy": _as_dict(_as_dict(w).get("empathy")).get("b"),
        }
        rows.append(row)

    if not rows:
        raise SystemExit("No valid trajectory.jsonl found under outdirs.\n" + "\n".join(errors))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Wrote: {out_csv} ({len(rows)} rows)")
    if errors:
        print("Warnings (skipped):")
        for e in errors[:50]:
            print("  -", e)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
