#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_TIKTOKEN_ENCODING = None


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of (approximated) tokens in a text string."""
    import tiktoken

    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None or getattr(_TIKTOKEN_ENCODING, "name", None) != encoding_name:
        _TIKTOKEN_ENCODING = tiktoken.get_encoding(encoding_name)
    num_tokens = len(_TIKTOKEN_ENCODING.encode(string, disallowed_special=()))
    return num_tokens


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _var_population(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _trajectory_overall_score(tr: Dict[str, Any]) -> Optional[float]:
    dims = tr.get("dimensions")
    if not isinstance(dims, dict):
        return None
    scores: List[float] = []
    for d in ["helpfulness", "trustfulness", "empathy"]:
        drec = dims.get(d)
        if not isinstance(drec, dict):
            continue
        s = _safe_float(drec.get("score"))
        if s is not None:
            scores.append(s)
    return _mean(scores)


def _find_outdirs(base: Path, pattern: str) -> List[Path]:
    return sorted([p for p in base.glob(pattern) if p.is_dir()])


def _load_model_summary(outdir: Path) -> Dict[str, Any]:
    js = outdir / "model_summary.json"
    if js.exists():
        return _read_json(js)
    # fallback: parse CSV into a dict with similar keys (best-effort)
    cs = outdir / "model_summary.csv"
    if cs.exists():
        import csv as _csv

        with cs.open("r", encoding="utf-8", newline="") as f:
            r = _csv.DictReader(f)
            row = next(r, None)
        if not row:
            raise ValueError(f"Empty model_summary.csv: {cs}")
        dims = {}
        for d in ["helpfulness", "trustfulness", "empathy"]:
            dims[d] = {"mean": _safe_float(row.get(f"{d}_mean"))}
        return {
            "n_trajectories": int(float(row.get("n_trajectories") or 0)),
            "dimensions": dims,
            "overall_of_dimension_means": _safe_float(row.get("overall_of_dimension_means")),
            "overall_per_trajectory_mean": {
                "mean": _safe_float(row.get("overall_per_trajectory_mean")),
                "std": _safe_float(row.get("overall_per_trajectory_std")),
                "n": int(float(row.get("overall_per_trajectory_n") or 0)),
            },
            "objective": {
                "total_env_steps": {"mean": _safe_float(row.get("total_env_steps_mean")), "std": _safe_float(row.get("total_env_steps_std"))},
                "finished_rate": {"mean": _safe_float(row.get("finished_rate_mean"))},
            },
        }
    raise FileNotFoundError(f"Missing model_summary.json/csv in {outdir}")


def _token_stats_and_by_task_for_outdir(outdir: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      (overall_stats, by_task_stats)
    Both based on window_text records.
    """
    windows = outdir / "windows.jsonl"
    detailed = outdir / "results_detailed.jsonl"

    n_windows = 0
    token_sum = 0
    task_sum: Dict[str, int] = {}
    task_n: Dict[str, int] = {}

    def consume(recs: Iterable[Dict[str, Any]], field: str) -> None:
        nonlocal n_windows, token_sum
        for r in recs:
            txt = r.get(field)
            if not isinstance(txt, str) or not txt:
                continue
            tks = num_tokens_from_string(txt)
            n_windows += 1
            token_sum += tks
            task = r.get("task_name")
            if isinstance(task, str) and task:
                task_n[task] = task_n.get(task, 0) + 1
                task_sum[task] = task_sum.get(task, 0) + tks

    if windows.exists():
        consume(_iter_jsonl(windows), "window_text")
    elif detailed.exists():
        consume(_iter_jsonl(detailed), "window_text")
    else:
        return (
            {
                "window_tokens_n": 0,
                "window_tokens_sum": None,
                "window_tokens_mean": None,
                "request_input_tokens_sum_est": None,
            },
            {},
        )

    # Add system prompt tokens (3 dims) as a rough input-token estimate.
    try:
        from llmjudge_prompts import EMPATHY_PROMPT, HELPFULNESS_PROMPT, TRUSTFULNESS_PROMPT  # type: ignore

        sys_tokens = (
            num_tokens_from_string(HELPFULNESS_PROMPT)
            + num_tokens_from_string(TRUSTFULNESS_PROMPT)
            + num_tokens_from_string(EMPATHY_PROMPT)
        )
    except Exception:
        sys_tokens = None

    request_sum_est = None
    if sys_tokens is not None and n_windows:
        request_sum_est = token_sum * 3 + sys_tokens * n_windows

    overall = {
        "window_tokens_n": n_windows,
        "window_tokens_sum": token_sum,
        "window_tokens_mean": (token_sum / n_windows) if n_windows else None,
        "request_input_tokens_sum_est": request_sum_est,
    }

    by_task: Dict[str, Dict[str, Any]] = {}
    for task, n in task_n.items():
        s = task_sum.get(task, 0)
        by_task[task] = {
            "task_window_tokens_n": n,
            "task_window_tokens_sum": s,
            "task_window_tokens_mean": (s / n) if n else None,
        }
    return overall, by_task


def _send_message_ratio_stats(outdir: Path, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For CoELA: uses objective agent_send_message_steps / agent_env_steps.
    For ProAgent: counts non-empty `message` in actions.json for the agent_id divided by len(entries).
    """
    agent_ratios: List[float] = []
    human_ratios: List[float] = []

    for tr in trajectories:
        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue

        # CoELA path
        a_send = _safe_float(obj.get("agent_send_message_steps"))
        a_env = _safe_float(obj.get("agent_env_steps"))
        if a_send is not None and a_env is not None and a_env > 0:
            agent_ratios.append(float(a_send) / float(a_env))

        h_send = _safe_float(obj.get("human_send_message_steps"))
        h_env = _safe_float(obj.get("human_env_steps"))
        if h_send is not None and h_env is not None and h_env > 0:
            human_ratios.append(float(h_send) / float(h_env))
        if (a_send is not None and a_env is not None) or (h_send is not None and h_env is not None):
            continue

        # ProAgent path (heavy, but number of trajectories is small)
        actions_json = obj.get("actions_json")
        agent_id = obj.get("agent_id")
        human_id = obj.get("human_id")
        if not isinstance(actions_json, str) or not actions_json:
            continue
        if agent_id is None:
            continue
        p = Path(actions_json)
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        for who, acc in [("agent", agent_ratios), ("human", human_ratios)]:
            wid = agent_id if who == "agent" else human_id
            if wid is None:
                continue
            entries = raw.get(str(int(wid)))
            if not isinstance(entries, list) or not entries:
                continue
            msg_n = 0
            for e in entries:
                if not isinstance(e, dict):
                    continue
                m = e.get("message")
                if isinstance(m, str) and m.strip():
                    msg_n += 1
            acc.append(msg_n / float(len(entries)))

    return {
        "agent_send_message_ratio_mean": _mean(agent_ratios),
        "agent_send_message_ratio_n": len(agent_ratios),
        "human_send_message_ratio_mean": _mean(human_ratios),
        "human_send_message_ratio_n": len(human_ratios),
    }


def _persona_variance_stats(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes: for each task, variance over 30 personas; then mean of task variances.
    Only applies when objective contains persona_id.
    """
    # task -> persona -> list[overall_score]
    task_persona_scores: Dict[str, Dict[int, List[float]]] = {}
    for tr in trajectories:
        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue
        persona = obj.get("persona_id")
        task = tr.get("task_name") or obj.get("task_name")
        if not isinstance(task, str) or not task:
            continue
        if persona is None:
            continue
        try:
            persona_id = int(persona)
        except Exception:
            continue
        s = _trajectory_overall_score(tr)
        if s is None:
            continue
        task_persona_scores.setdefault(task, {}).setdefault(persona_id, []).append(s)

    task_vars: List[float] = []
    task_persona_counts: List[int] = []
    for task, persona_map in task_persona_scores.items():
        persona_means: List[float] = []
        for _, scores in persona_map.items():
            m = _mean(scores)
            if m is not None:
                persona_means.append(m)
        if persona_means:
            v = _var_population(persona_means)
            if v is not None:
                task_vars.append(v)
                task_persona_counts.append(len(persona_means))

    return {
        "task_persona_variance_mean": _mean(task_vars),
        "task_persona_variance_tasks": len(task_vars),
        "task_persona_persona_count_mean": _mean([float(x) for x in task_persona_counts]) if task_persona_counts else None,
    }


def _token_stats_by_task_for_outdir(outdir: Path) -> Dict[str, Dict[str, Any]]:
    _, by_task = _token_stats_and_by_task_for_outdir(outdir)
    return by_task


def _persona_variance_for_task(trajs: List[Dict[str, Any]], task: str) -> Optional[float]:
    # persona -> list[overall_score]
    persona_scores: Dict[int, List[float]] = {}
    for tr in trajs:
        if str(tr.get("task_name") or "") != task:
            continue
        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue
        persona = obj.get("persona_id")
        if persona is None:
            continue
        try:
            persona_id = int(persona)
        except Exception:
            continue
        s = _trajectory_overall_score(tr)
        if s is None:
            continue
        persona_scores.setdefault(persona_id, []).append(s)

    persona_means: List[float] = []
    for scores in persona_scores.values():
        m = _mean(scores)
        if m is not None:
            persona_means.append(m)
    return _var_population(persona_means)


def _send_message_ratios_for_task(outdir: Path, trajectories: List[Dict[str, Any]], task: str) -> Tuple[Optional[float], Optional[float], int]:
    agent_ratios: List[float] = []
    human_ratios: List[float] = []
    n = 0
    for tr in trajectories:
        if str(tr.get("task_name") or "") != task:
            continue
        n += 1
        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue

        a_send = _safe_float(obj.get("agent_send_message_steps"))
        a_env = _safe_float(obj.get("agent_env_steps"))
        if a_send is not None and a_env is not None and a_env > 0:
            agent_ratios.append(float(a_send) / float(a_env))

        h_send = _safe_float(obj.get("human_send_message_steps"))
        h_env = _safe_float(obj.get("human_env_steps"))
        if h_send is not None and h_env is not None and h_env > 0:
            human_ratios.append(float(h_send) / float(h_env))

        if (a_send is not None and a_env is not None) or (h_send is not None and h_env is not None):
            continue

        actions_json = obj.get("actions_json")
        agent_id = obj.get("agent_id")
        human_id = obj.get("human_id")
        if not isinstance(actions_json, str) or not actions_json:
            continue
        if agent_id is None:
            continue
        p = Path(actions_json)
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue

        for who, acc in [("agent", agent_ratios), ("human", human_ratios)]:
            wid = agent_id if who == "agent" else human_id
            if wid is None:
                continue
            entries = raw.get(str(int(wid)))
            if not isinstance(entries, list) or not entries:
                continue
            msg_n = 0
            for e in entries:
                if not isinstance(e, dict):
                    continue
                m = e.get("message")
                if isinstance(m, str) and m.strip():
                    msg_n += 1
            acc.append(msg_n / float(len(entries)))

    return _mean(agent_ratios), _mean(human_ratios), n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Final comparison report across multiple out_* folders (adds tokens/variance/zscore/sigmoid/send_message ratio).")
    ap.add_argument("--bases", nargs="+", required=True, help="Base directories containing out_* folders.")
    ap.add_argument("--glob", default="out_*", help="Glob pattern under each base.")
    ap.add_argument("--out-csv", required=True, help="Output CSV path.")
    ap.add_argument("--out-task-csv", default=None, help="Optional per-task CSV path (one row per model per task).")
    ap.add_argument("--include-outdir", action="store_true", help="Include absolute outdir path.")
    args = ap.parse_args(argv)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    outdirs: List[Tuple[str, Path]] = []
    for b in args.bases:
        base = Path(b).expanduser()
        if not base.is_absolute():
            base = (Path.cwd() / base).resolve()
        for d in _find_outdirs(base, args.glob):
            outdirs.append((str(base), d))

    rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for base_str, outdir in outdirs:
        try:
            ms = _load_model_summary(outdir)
            traj_path = outdir / "trajectory.jsonl"
            trajectories = list(_iter_jsonl(traj_path)) if traj_path.exists() else []
            token_overall, tokens_by_task = _token_stats_and_by_task_for_outdir(outdir)

            dims = ms.get("dimensions") if isinstance(ms.get("dimensions"), dict) else {}
            row: Dict[str, Any] = {
                "base": base_str,
                "name": outdir.name,
                "n_trajectories": ms.get("n_trajectories"),
                "helpfulness_mean": (dims.get("helpfulness") or {}).get("mean") if isinstance(dims.get("helpfulness"), dict) else None,
                "trustfulness_mean": (dims.get("trustfulness") or {}).get("mean") if isinstance(dims.get("trustfulness"), dict) else None,
                "empathy_mean": (dims.get("empathy") or {}).get("mean") if isinstance(dims.get("empathy"), dict) else None,
                "overall_of_dimension_means": ms.get("overall_of_dimension_means"),
            }
            overall_traj = ms.get("overall_per_trajectory_mean") if isinstance(ms.get("overall_per_trajectory_mean"), dict) else {}
            row["overall_per_trajectory_mean"] = overall_traj.get("mean") if isinstance(overall_traj, dict) else None
            row["overall_per_trajectory_std"] = overall_traj.get("std") if isinstance(overall_traj, dict) else None

            obj = ms.get("objective") if isinstance(ms.get("objective"), dict) else {}
            total_steps = obj.get("total_env_steps") if isinstance(obj.get("total_env_steps"), dict) else {}
            finished = obj.get("finished_rate") if isinstance(obj.get("finished_rate"), dict) else {}
            row["total_env_steps_mean"] = total_steps.get("mean") if isinstance(total_steps, dict) else None
            row["finished_rate_mean"] = finished.get("mean") if isinstance(finished, dict) else None

            # Added stats
            row.update(token_overall)
            row.update(_persona_variance_stats(trajectories))
            row.update(_send_message_ratio_stats(outdir, trajectories))

            if args.include_outdir:
                row["outdir"] = str(outdir)
            rows.append(row)

            if args.out_task_csv and trajectories:
                # Build task-level aggregates for this outdir.
                tasks = sorted({str(t.get("task_name")) for t in trajectories if isinstance(t.get("task_name"), str) and t.get("task_name")})
                for task in tasks:
                    t_trajs = [t for t in trajectories if str(t.get("task_name") or "") == task]
                    overall_scores: List[float] = []
                    dim_scores: Dict[str, List[float]] = {"helpfulness": [], "trustfulness": [], "empathy": []}
                    persona_ids: set[int] = set()
                    for tr in t_trajs:
                        s = _trajectory_overall_score(tr)
                        if s is not None:
                            overall_scores.append(s)
                        drec = tr.get("dimensions")
                        if isinstance(drec, dict):
                            for d in ["helpfulness", "trustfulness", "empathy"]:
                                dd = drec.get(d)
                                if isinstance(dd, dict):
                                    sv = _safe_float(dd.get("score"))
                                    if sv is not None:
                                        dim_scores[d].append(sv)
                        obj = tr.get("objective")
                        if isinstance(obj, dict) and obj.get("persona_id") is not None:
                            try:
                                persona_ids.add(int(obj.get("persona_id")))
                            except Exception:
                                pass

                    agent_ratio, human_ratio, tn = _send_message_ratios_for_task(outdir, trajectories, task)
                    var_task = _persona_variance_for_task(trajectories, task)

                    trow: Dict[str, Any] = {
                        "base": base_str,
                        "name": outdir.name,
                        "task_name": task,
                        "n_trajectories": tn,
                        "n_personas": len(persona_ids) if persona_ids else None,
                        "overall_mean": _mean(overall_scores),
                        "helpfulness_mean": _mean(dim_scores["helpfulness"]),
                        "trustfulness_mean": _mean(dim_scores["trustfulness"]),
                        "empathy_mean": _mean(dim_scores["empathy"]),
                        "persona_variance": var_task,
                        "agent_send_message_ratio_mean": agent_ratio,
                        "human_send_message_ratio_mean": human_ratio,
                    }
                    if task in tokens_by_task:
                        trow.update(tokens_by_task[task])
                    if args.include_outdir:
                        trow["outdir"] = str(outdir)
                    task_rows.append(trow)
        except Exception as e:
            errors.append(f"{outdir}: {e}")

    if not rows:
        raise SystemExit("No valid out_* folders found.\n" + "\n".join(errors))

    # Z-score + sigmoid on subjective metric across rows
    subj_vals: List[float] = []
    for r in rows:
        v = _safe_float(r.get("overall_of_dimension_means"))
        if v is not None:
            subj_vals.append(v)

    subj_mean = _mean(subj_vals) or 0.0
    subj_std = math.sqrt(_var_population(subj_vals) or 0.0) or 0.0
    for r in rows:
        v = _safe_float(r.get("overall_of_dimension_means"))
        if v is None or subj_std == 0.0:
            r["subjective_zscore"] = None
            r["subjective_sigmoid"] = None
        else:
            z = (v - subj_mean) / subj_std
            r["subjective_zscore"] = z
            r["subjective_sigmoid"] = _sigmoid(z)

    fieldnames = [
        "base",
        "name",
        "n_trajectories",
        "finished_rate_mean",
        "total_env_steps_mean",
        "helpfulness_mean",
        "trustfulness_mean",
        "empathy_mean",
        "overall_of_dimension_means",
        "overall_per_trajectory_mean",
        "overall_per_trajectory_std",
        "subjective_zscore",
        "subjective_sigmoid",
        "agent_send_message_ratio_mean",
        "agent_send_message_ratio_n",
        "human_send_message_ratio_mean",
        "human_send_message_ratio_n",
        "task_persona_variance_mean",
        "task_persona_variance_tasks",
        "task_persona_persona_count_mean",
        "window_tokens_n",
        "window_tokens_sum",
        "window_tokens_mean",
        "request_input_tokens_sum_est",
    ]
    if args.include_outdir:
        fieldnames.append("outdir")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    if args.out_task_csv:
        task_csv = Path(args.out_task_csv).expanduser().resolve()
        task_csv.parent.mkdir(parents=True, exist_ok=True)

        # Per-task zscore+sigmoid across models (within each task)
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for r in task_rows:
            t = r.get("task_name")
            if isinstance(t, str) and t:
                by_task.setdefault(t, []).append(r)
        for t, rs in by_task.items():
            vals: List[float] = []
            for r in rs:
                v = _safe_float(r.get("overall_mean"))
                if v is not None:
                    vals.append(v)
            m = _mean(vals) or 0.0
            std = math.sqrt(_var_population(vals) or 0.0) or 0.0
            for r in rs:
                v = _safe_float(r.get("overall_mean"))
                if v is None or std == 0.0:
                    r["overall_zscore_in_task"] = None
                    r["overall_sigmoid_in_task"] = None
                else:
                    z = (v - m) / std
                    r["overall_zscore_in_task"] = z
                    r["overall_sigmoid_in_task"] = _sigmoid(z)

        task_fieldnames = [
            "base",
            "name",
            "task_name",
            "n_trajectories",
            "n_personas",
            "overall_mean",
            "helpfulness_mean",
            "trustfulness_mean",
            "empathy_mean",
            "overall_zscore_in_task",
            "overall_sigmoid_in_task",
            "persona_variance",
            "agent_send_message_ratio_mean",
            "human_send_message_ratio_mean",
            "task_window_tokens_n",
            "task_window_tokens_sum",
            "task_window_tokens_mean",
        ]
        if args.include_outdir:
            task_fieldnames.append("outdir")

        with task_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=task_fieldnames)
            w.writeheader()
            for r in task_rows:
                w.writerow({k: r.get(k) for k in task_fieldnames})
        print(f"Wrote: {task_csv} ({len(task_rows)} rows)")

    if errors:
        print("Warnings (skipped):")
        for msg in errors:
            print("  -", msg)
    print(f"Wrote: {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
