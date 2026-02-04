#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import pickle
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from llmjudge_format import format_window_for_judge, format_window_preview
from llmjudge_llm import judge_one_dimension, openai_client
from llmjudge_logio import iter_log_paths, load_log
from llmjudge_prompts import EMPATHY_PROMPT, HELPFULNESS_PROMPT, TRUSTFULNESS_PROMPT
from llmjudge_windowing import build_windows_from_log


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    _append_jsonl(path, obj)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is not None and val.strip() != "":
        return val
    return fallback


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float_opt(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# =========================
# build-windows
# =========================

def _count_action(actions: Any, token: str) -> int:
    if not isinstance(actions, list):
        return 0
    return sum(1 for a in actions if isinstance(a, str) and token in a)


def _objective_metrics(log: Dict[str, Any], agent_id: int, human_id: int) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

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

    metrics["total_env_steps"] = total_env_steps
    metrics["agent_env_steps"] = agent_env_steps
    metrics["human_env_steps"] = human_env_steps

    metrics["agent_send_message_steps"] = _count_action(agent_actions, "[send_message]")
    metrics["human_send_message_steps"] = _count_action(human_actions, "[send_message]")

    llm = log.get("LLM", {})
    metrics["agent_llm_calls"] = len(llm.get(agent_id, [])) if isinstance(llm, dict) else None
    metrics["human_llm_calls"] = len(llm.get(human_id, [])) if isinstance(llm, dict) else None

    finished = log.get("finished")
    if isinstance(finished, bool):
        metrics["finished"] = finished
    metrics["task_success"] = bool(finished) if finished is not None else None

    if isinstance(log.get("goals_finished"), list):
        metrics["goals_finished_len"] = len(log["goals_finished"])

    return metrics


def cmd_build_windows(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export cut windows from CoELA/CWAH logs (window=3 decisions).")
    ap.add_argument("--input", required=True, help="Run dir, runs root, or a single `logs_agent_*.pik` file.")
    ap.add_argument("--out", required=True, help="Output windows JSONL (one line per window).")
    ap.add_argument("--preview-out", default=None, help="Optional path to save a human-readable preview text.")
    ap.add_argument("--agent-id", type=int, required=True, help="Agent id under evaluation (assistant).")
    ap.add_argument("--human-id", type=int, required=True, help="Partner id (human).")
    ap.add_argument("--window-size", type=int, default=3, help="Window size (must be 3).")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of log files processed.")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=2200,
        help="Truncate each field to this many chars to control window_text size.",
    )
    ap.add_argument("--preview-max-chars", type=int, default=0, help="Truncate preview text; 0 means no truncation.")

    args = ap.parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preview_file = None
    if args.preview_out:
        preview_path = Path(args.preview_out).expanduser().resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_file = preview_path.open("w", encoding="utf-8")

    log_paths = iter_log_paths(input_path, args.max_files)
    if not log_paths:
        raise SystemExit(f"No `logs_agent_*.pik` found under: {input_path}")

    run_ts = _dt.datetime.now().isoformat(timespec="seconds")

    for pik_path in log_paths:
        log = load_log(pik_path)
        if log is None:
            print(f"[warn] skip unreadable log: {pik_path}")
            continue
        run_dir = str(pik_path.parent)
        log_path = str(pik_path)

        variant_root = pik_path.parents[2]
        personality_json = variant_root / "LLM" / "data_cleaned3_enriched.json"
        fallback_human_personality = None
        m = re.search(r"_([0-9]+)_task[0-9]+$", Path(run_dir).name)
        cluster_id = m.group(1) if m else None
        if personality_json.exists() and cluster_id is not None:
            try:
                data = json.loads(personality_json.read_text(encoding="utf-8"))
                profile_dict = {}
                for item in data if isinstance(data, list) else []:
                    try:
                        key = f"{item.get('Task')}_{item.get('Cluster')}"
                        profile_dict[key] = item.get("Profile", "")
                    except Exception:
                        continue
                task_name = str(log.get("task_name", ""))
                fallback_human_personality = profile_dict.get(f"{task_name}_{cluster_id}") or None
            except Exception:
                fallback_human_personality = None

        windows = build_windows_from_log(
            log,
            run_dir=run_dir,
            log_path=log_path,
            agent_id=args.agent_id,
            oppo_id=args.human_id,
            fallback_oppo_personality=fallback_human_personality,
            window_size=args.window_size,
        )

        obj = _objective_metrics(log, args.agent_id, args.human_id)
        header = (
            f"# Windows: {log_path} agent_id={args.agent_id} human_id={args.human_id} windows={len(windows)}"
        )
        if preview_file:
            preview_file.write(header + "\n")

        for w in windows:
            window_text = format_window_for_judge(w, args.max_chars)
            run_tag = Path(run_dir).name
            window_id = (
                f"{run_tag}::{Path(log_path).name}::agent{w.agent_id}::human{args.human_id}::win{w.window_index}"
            )
            rec: Dict[str, Any] = {
                "run_ts": run_ts,
                "window_id": window_id,
                "run_dir": w.run_dir,
                "log_path": w.log_path,
                "task_id": w.task_id,
                "env_id": w.env_id,
                "task_name": w.task_name,
                "agent_id": w.agent_id,
                "human_id": args.human_id,
                "window_index": w.window_index,
                "llm_indices": [s.llm_index for s in w.steps],
                "step_nows": [s.step_now for s in w.steps],
                "agent_name": w.steps[0].agent_name,
                "human_name": w.steps[0].oppo_name,
                "window_text": window_text,
                "objective": obj,
            }
            _append_jsonl(out_path, rec)

            if preview_file:
                preview_file.write(format_window_preview(w, max_chars=args.preview_max_chars) + "\n")

    if preview_file:
        preview_file.close()
    return 0


# =========================
# build-prompts
# =========================

def cmd_build_prompts(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build filled LLM-judge prompts from windows JSONL.")
    ap.add_argument("--windows", required=True, help="Input windows JSONL (from build-windows).")
    ap.add_argument("--out", required=True, help="Output prompts JSONL (one line per request).")
    ap.add_argument("--thin", action="store_true", help="Write compact requests (no prompt text), keyed by window_id.")
    ap.add_argument(
        "--include-window-text",
        action="store_true",
        help="Store window_text inside each record (duplicates user_prompt). Ignored in --thin.",
    )

    args = ap.parse_args(argv)
    windows_path = Path(args.windows).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_ts = _dt.datetime.now().isoformat(timespec="seconds")
    prompts = {
        "helpfulness": HELPFULNESS_PROMPT,
        "trustfulness": TRUSTFULNESS_PROMPT,
        "empathy": EMPATHY_PROMPT,
    }

    for wrec in _iter_jsonl(windows_path):
        window_id = wrec.get("window_id")
        window_text = wrec.get("window_text", "")
        base: Dict[str, Any] = {
            "run_ts": run_ts,
            "window_id": window_id,
            "run_dir": wrec.get("run_dir"),
            "log_path": wrec.get("log_path"),
            "task_id": wrec.get("task_id"),
            "env_id": wrec.get("env_id"),
            "task_name": wrec.get("task_name"),
            "agent_id": wrec.get("agent_id"),
            "human_id": wrec.get("human_id"),
            "window_index": wrec.get("window_index"),
            "llm_indices": wrec.get("llm_indices"),
            "step_nows": wrec.get("step_nows"),
            "agent_name": wrec.get("agent_name"),
            "human_name": wrec.get("human_name"),
            "objective": wrec.get("objective"),
        }
        if (not args.thin) and args.include_window_text:
            base["window_text"] = window_text

        for dim, sys_prompt in prompts.items():
            rec = dict(base)
            rec["dimension"] = dim
            if not args.thin:
                rec["system_prompt"] = sys_prompt
                rec["user_prompt"] = window_text
            rec["request_id"] = f"{window_id}::{dim}"
            _write_jsonl(out_path, rec)

    return 0


# =========================
# score-prompts
# =========================

def cmd_score_prompts(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score filled prompts JSONL and write results JSONL.")
    ap.add_argument("--prompts", required=True, help="Input prompts JSONL (from build-prompts). Can be --thin.")
    ap.add_argument("--windows", default=None, help="Windows JSONL (required if --prompts was built with --thin).")
    ap.add_argument("--out", required=True, help="Output results JSONL (one line per prompt record).")
    ap.add_argument("--resume", action="store_true", help="Skip request_ids already present in --out.")
    ap.add_argument("--lite", action="store_true", default=True, help="Write compact results (drop large fields).")
    ap.add_argument("--keep-input", action="store_true", help="Keep input prompt fields in results (overrides --lite).")

    ap.add_argument("--model", default=_default_env("OPENAI_MODEL", "gpt-4o-mini"), help="Judge model name.")
    ap.add_argument(
        "--api-base",
        default=_default_env("OPENAI_API_BASE") or _default_env("OPENAI_BASE_URL") or "http://localhost:8000/v1",
        help="OpenAI-compatible base_url (should include /v1).",
    )
    ap.add_argument("--api-key", default=_default_env("OPENAI_API_KEY", "EMPTY"), help="API key.")
    ap.add_argument("--max-tokens", type=int, default=900, help="Max tokens per judge call.")
    ap.add_argument("--retries", type=int, default=3, help="Retries per judge call for JSON parsing/robustness.")
    ap.add_argument("--limit", type=int, default=None, help="Only score first N prompts (for testing).")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")

    args = ap.parse_args(argv)
    prompts_path = Path(args.prompts).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    windows_map: Dict[str, str] = {}
    windows_obj: Dict[str, Any] = {}
    if args.windows:
        wpath = Path(args.windows).expanduser().resolve()
        for w in _iter_jsonl(wpath):
            wid = w.get("window_id")
            wtxt = w.get("window_text")
            if isinstance(wid, str) and wid and isinstance(wtxt, str):
                windows_map[wid] = wtxt
                if "objective" in w:
                    windows_obj[wid] = w.get("objective")

    done = set()
    if args.resume and out_path.exists():
        for rec in _iter_jsonl(out_path):
            rid = rec.get("request_id")
            if isinstance(rid, str) and rid:
                done.add(rid)

    run_ts = _dt.datetime.now().isoformat(timespec="seconds")

    work: list[Dict[str, Any]] = []
    for rec in _iter_jsonl(prompts_path):
        rid = rec.get("request_id")
        if args.resume and isinstance(rid, str) and rid in done:
            continue
        work.append(rec)
        if args.limit is not None and len(work) >= args.limit:
            break

    lock = threading.Lock()

    dim_to_prompt = {
        "helpfulness": HELPFULNESS_PROMPT,
        "trustfulness": TRUSTFULNESS_PROMPT,
        "empathy": EMPATHY_PROMPT,
    }

    def _score_one(rec: Dict[str, Any]) -> Dict[str, Any]:
        client = openai_client(args.api_key, args.api_base)
        dim = str(rec.get("dimension", "")).strip()
        system_prompt = rec.get("system_prompt", "") or dim_to_prompt.get(dim, "")
        user_prompt = rec.get("user_prompt", "")
        if (not isinstance(user_prompt, str) or user_prompt == ""):
            wid = rec.get("window_id")
            if isinstance(wid, str) and wid in windows_map:
                user_prompt = windows_map[wid]
        if not isinstance(user_prompt, str) or user_prompt == "":
            raise RuntimeError(
                "Missing user_prompt; provide full prompts.jsonl or pass --windows when using --thin prompts."
            )

        score_obj = judge_one_dimension(
            client=client,
            model=args.model,
            dimension_prompt=str(system_prompt),
            window_text=str(user_prompt),
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
        out_rec = dict(rec)
        out_rec["score_run_ts"] = run_ts
        out_rec["model"] = args.model
        out_rec["result"] = score_obj
        wid = out_rec.get("window_id")
        if isinstance(wid, str) and wid in windows_obj and "objective" not in out_rec:
            out_rec["objective"] = windows_obj[wid]
        if args.lite and (not args.keep_input):
            for k in ["system_prompt", "user_prompt", "window_text"]:
                if k in out_rec:
                    out_rec.pop(k, None)
        return out_rec

    def _write_one(out_rec: Dict[str, Any]) -> None:
        with lock:
            _append_jsonl(out_path, out_rec)

    if args.workers <= 1:
        for rec in work:
            rid = rec.get("request_id")
            try:
                out_rec = _score_one(rec)
            except Exception as e:
                out_rec = dict(rec)
                out_rec["score_run_ts"] = run_ts
                out_rec["model"] = args.model
                out_rec["error"] = str(e)
            _write_one(out_rec)
            if isinstance(rid, str) and rid:
                done.add(rid)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_score_one, rec): rec for rec in work}
            for fut in as_completed(futures):
                rec = futures[fut]
                rid = rec.get("request_id")
                try:
                    out_rec = fut.result()
                except Exception as e:
                    out_rec = dict(rec)
                    out_rec["score_run_ts"] = run_ts
                    out_rec["model"] = args.model
                    out_rec["error"] = str(e)
                _write_one(out_rec)
                if isinstance(rid, str) and rid:
                    done.add(rid)

    return 0


# =========================
# backfill-objective
# =========================

def cmd_backfill_objective(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill objective metrics into results JSONL using windows.jsonl.")
    ap.add_argument("--windows", required=True, help="windows.jsonl (must contain objective per window).")
    ap.add_argument("--results", required=True, help="results_lite.jsonl (may be missing objective).")
    ap.add_argument("--out", required=True, help="Output results JSONL with objective filled in.")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting --out if it exists.")
    args = ap.parse_args(argv)

    windows_path = Path(args.windows).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path} (use --overwrite)")

    win_obj: Dict[str, Any] = {}
    for w in _iter_jsonl(windows_path):
        wid = w.get("window_id")
        obj = w.get("objective")
        if isinstance(wid, str) and wid and isinstance(obj, dict):
            win_obj[wid] = obj

    if out_path.exists():
        out_path.unlink()

    n = 0
    filled = 0
    for r in _iter_jsonl(results_path):
        n += 1
        if "objective" not in r:
            wid = r.get("window_id")
            if isinstance(wid, str) and wid in win_obj:
                r["objective"] = win_obj[wid]
                filled += 1
        _append_jsonl(out_path, r)

    print(f"backfill done: total={n} filled={filled} out={out_path}")
    return 0


# =========================
# merge-outputs
# =========================

def _violation_rules(res: Any) -> List[str]:
    if not isinstance(res, dict):
        return []
    violations = res.get("violations", [])
    if not isinstance(violations, list):
        return []
    rules = []
    for v in violations:
        if not isinstance(v, dict):
            continue
        rule = str(v.get("rule", "")).strip()
        if rule:
            rules.append(rule)
    return rules


def _violation_rule_counts(res: Any) -> Dict[str, int]:
    return dict(Counter(_violation_rules(res)))


def _strip_to_compact_viols(res: Any) -> Dict[str, Any]:
    if not isinstance(res, dict):
        return {"score": None, "rules": []}
    score = res.get("score")
    rules = _violation_rules(res)
    return {"score": score, "rules": sorted(set(rules))}


def cmd_merge_outputs(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Merge process files into detailed and compact results.")
    ap.add_argument("--windows", required=True, help="windows.jsonl from build-windows")
    ap.add_argument("--results", required=True, help="results.jsonl/results_lite.jsonl from score-prompts")
    ap.add_argument("--out-detailed", required=True, help="Output detailed results JSONL (one line per window).")
    ap.add_argument("--out-compact", required=True, help="Output compact results JSONL (one line per window).")
    ap.add_argument("--strict", action="store_true", help="If set, require all 3 dimensions exist per window.")

    args = ap.parse_args(argv)
    windows_path = Path(args.windows).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    out_detailed = Path(args.out_detailed).expanduser().resolve()
    out_compact = Path(args.out_compact).expanduser().resolve()
    out_detailed.parent.mkdir(parents=True, exist_ok=True)
    out_compact.parent.mkdir(parents=True, exist_ok=True)

    windows: Dict[str, Dict[str, Any]] = {}
    for w in _iter_jsonl(windows_path):
        wid = w.get("window_id")
        if isinstance(wid, str) and wid:
            windows[wid] = w

    per_window: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    meta_by_window: Dict[str, Dict[str, Any]] = {}
    for r in _iter_jsonl(results_path):
        wid = str(r.get("window_id", "")).strip()
        if not wid:
            rid = str(r.get("request_id", "")).strip()
            if "::" in rid:
                wid = "::".join(rid.split("::")[:-1])
        if not wid:
            continue
        dim = str(r.get("dimension", "")).strip()
        res = r.get("result", {})
        if dim:
            per_window[wid][dim] = res if isinstance(res, dict) else {}
        if wid not in meta_by_window:
            meta_by_window[wid] = {
                "model": r.get("model"),
                "score_run_ts": r.get("score_run_ts"),
                "agent_id": r.get("agent_id"),
                "human_id": r.get("human_id"),
                "log_path": r.get("log_path"),
                "run_dir": r.get("run_dir"),
                "task_id": r.get("task_id"),
                "env_id": r.get("env_id"),
                "task_name": r.get("task_name"),
                "window_index": r.get("window_index"),
            }

    if out_detailed.exists():
        out_detailed.unlink()
    if out_compact.exists():
        out_compact.unlink()

    for wid, dims in per_window.items():
        if args.strict:
            if not all(k in dims for k in ("helpfulness", "trustfulness", "empathy")):
                continue

        w = windows.get(wid, {})
        meta = meta_by_window.get(wid, {})

        detailed = {
            "window_id": wid,
            "meta": {
                **{k: meta.get(k) for k in meta.keys()},
                **{
                    k: w.get(k)
                    for k in [
                        "run_dir",
                        "log_path",
                        "task_id",
                        "env_id",
                        "task_name",
                        "agent_id",
                        "human_id",
                        "window_index",
                        "llm_indices",
                        "step_nows",
                        "agent_name",
                        "human_name",
                    ]
                },
            },
            "window_text": w.get("window_text"),
            "objective": w.get("objective"),
            "dimensions": dims,
        }
        _append_jsonl(out_detailed, detailed)

        compact = {
            "window_id": wid,
            "log_path": meta.get("log_path") or w.get("log_path"),
            "task_id": meta.get("task_id") or w.get("task_id"),
            "env_id": meta.get("env_id") or w.get("env_id"),
            "task_name": meta.get("task_name") or w.get("task_name"),
            "agent_id": _safe_int(meta.get("agent_id") or w.get("agent_id"), -1),
            "human_id": _safe_int(meta.get("human_id") or w.get("human_id"), -1),
            "window_index": meta.get("window_index") or w.get("window_index"),
            "objective": w.get("objective"),
            "helpfulness": _strip_to_compact_viols(dims.get("helpfulness")),
            "trustfulness": _strip_to_compact_viols(dims.get("trustfulness")),
            "empathy": _strip_to_compact_viols(dims.get("empathy")),
        }
        compact["rule_counts"] = {
            "helpfulness": _violation_rule_counts(dims.get("helpfulness")),
            "trustfulness": _violation_rule_counts(dims.get("trustfulness")),
            "empathy": _violation_rule_counts(dims.get("empathy")),
        }
        _append_jsonl(out_compact, compact)

    return 0


# =========================
# export-window-csv
# =========================

def _viol_count(res: Any) -> int:
    if not isinstance(res, dict):
        return 0
    v = res.get("violations", [])
    return len(v) if isinstance(v, list) else 0


def _ded_sum(res: Any) -> float:
    if not isinstance(res, dict):
        return 0.0
    v = res.get("violations", [])
    if not isinstance(v, list):
        return 0.0
    s = 0.0
    for item in v:
        if isinstance(item, dict):
            s += _safe_float(item.get("deduction"), 0.0)
    return s


def cmd_export_window_csv(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export window-level results to a compact CSV for analysis.")
    ap.add_argument("--results", required=True, help="Input results JSONL (from score-prompts).")
    ap.add_argument("--out-csv", required=True, help="Output CSV path.")
    args = ap.parse_args(argv)

    results_path = Path(args.results).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    table: Dict[Tuple[str, int, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    meta: Dict[Tuple[str, int, int, str], Dict[str, Any]] = {}

    for rec in _iter_jsonl(results_path):
        log_path = str(rec.get("log_path", ""))
        agent_id = _safe_int(rec.get("agent_id"), -1)
        human_id = _safe_int(rec.get("human_id"), -1)
        window_id = str(rec.get("window_id", "")) or str(rec.get("request_id", ""))
        dim = str(rec.get("dimension", "")).strip()
        res = rec.get("result", {})
        key = (log_path, agent_id, human_id, window_id)
        if key not in meta:
            meta[key] = {
                "task_id": rec.get("task_id"),
                "env_id": rec.get("env_id"),
                "task_name": rec.get("task_name"),
                "window_index": rec.get("window_index"),
                "agent_name": rec.get("agent_name"),
                "human_name": rec.get("human_name") or rec.get("oppo_name"),
                "objective": rec.get("objective"),
            }
        if dim:
            table[key][dim] = res if isinstance(res, dict) else {}

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "log_path",
                "task_id",
                "env_id",
                "task_name",
                "agent_id",
                "human_id",
                "agent_name",
                "human_name",
                "window_id",
                "window_index",
                "total_env_steps",
                "agent_env_steps",
                "human_env_steps",
                "agent_send_message_steps",
                "human_send_message_steps",
                "finished",
                "helpfulness_score",
                "trustfulness_score",
                "empathy_score",
                "helpfulness_violation_count",
                "trustfulness_violation_count",
                "empathy_violation_count",
                "helpfulness_deduction_sum",
                "trustfulness_deduction_sum",
                "empathy_deduction_sum",
            ],
        )
        w.writeheader()

        for key, dims in table.items():
            log_path, agent_id, human_id, window_id = key
            m = meta.get(key, {})
            obj = m.get("objective") if isinstance(m.get("objective"), dict) else {}
            row = {
                "log_path": log_path,
                "task_id": m.get("task_id"),
                "env_id": m.get("env_id"),
                "task_name": m.get("task_name"),
                "agent_id": agent_id,
                "human_id": human_id,
                "agent_name": m.get("agent_name"),
                "human_name": m.get("human_name"),
                "window_id": window_id,
                "window_index": m.get("window_index"),
                "total_env_steps": obj.get("total_env_steps") if isinstance(obj, dict) else "",
                "agent_env_steps": obj.get("agent_env_steps") if isinstance(obj, dict) else "",
                "human_env_steps": obj.get("human_env_steps") if isinstance(obj, dict) else "",
                "agent_send_message_steps": obj.get("agent_send_message_steps") if isinstance(obj, dict) else "",
                "human_send_message_steps": obj.get("human_send_message_steps") if isinstance(obj, dict) else "",
                "finished": obj.get("finished") if isinstance(obj, dict) else "",
            }
            for dim in ["helpfulness", "trustfulness", "empathy"]:
                res = dims.get(dim, {})
                row[f"{dim}_score"] = res.get("score") if isinstance(res, dict) else ""
                row[f"{dim}_violation_count"] = _viol_count(res)
                row[f"{dim}_deduction_sum"] = _ded_sum(res)
            w.writerow(row)

    return 0


# =========================
# aggregate-results
# =========================

def _deduction_sum(violations: Any) -> float:
    if not isinstance(violations, list):
        return 0.0
    s = 0.0
    for v in violations:
        if not isinstance(v, dict):
            continue
        s += _safe_float(v.get("deduction"), 0.0)
    return s


def _violation_count(violations: Any) -> int:
    return len(violations) if isinstance(violations, list) else 0


def _top_rules(violations_list: List[Any], topk: int) -> List[Dict[str, Any]]:
    c: Counter[str] = Counter()
    for violations in violations_list:
        if not isinstance(violations, list):
            continue
        for v in violations:
            if not isinstance(v, dict):
                continue
            rule = str(v.get("rule", "")).strip()
            if rule:
                c[rule] += 1
    return [{"rule": r, "count": n} for r, n in c.most_common(topk)]


_PROAGENT_ACTIONS_INDEX: Optional[Dict[str, Path]] = None


def _get_proagent_root() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    p = repo_root / "ProAgent_1221"
    return p if p.exists() else None


def _build_proagent_actions_index() -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    root = _get_proagent_root()
    if not root:
        return idx
    for p in root.glob("src-*/actions/*/step_1/actions.json"):
        actions_folder = p.parent.parent.name
        if actions_folder not in idx:
            idx[actions_folder] = p
    return idx


def _num_tokens(text: str, *, encoding_name: str = "cl100k_base") -> Tuple[int, str]:
    text = text or ""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text, disallowed_special=())), f"tiktoken:{encoding_name}"
    except Exception:
        return max(0, int(math.ceil(len(text) / 4.0))), "approx:len/4"


def _extract_output_text(llm_entry: Any) -> Optional[str]:
    if not isinstance(llm_entry, dict):
        return None
    for k in ["raw_output", "outputs"]:
        v = llm_entry.get(k)
        if isinstance(v, str) and v.strip():
            return v
    v = llm_entry.get("raw_outputs")
    if isinstance(v, list) and v:
        parts = [str(x) for x in v if x is not None]
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    v = llm_entry.get("message")
    if isinstance(v, str) and v.strip():
        return v
    return None


def _output_token_stats_for_trajectory(
    *, log_path: str, objective: Any, agent_id: int
) -> Dict[str, Any]:
    toks: List[int] = []
    tokenizer: Optional[str] = None
    send_message_ratio_llm: Optional[float] = None
    send_message_ratio_env: Optional[float] = None
    env_steps: Optional[int] = None
    llm_calls_total: Optional[int] = None
    llm_send_message_count: Optional[int] = None

    p = Path(log_path)

    if log_path.endswith(".pik") and p.exists():
        try:
            log = pickle.loads(p.read_bytes())
        except Exception:
            log = None
        if isinstance(log, dict):
            llm = log.get("LLM", {})
            entries = llm.get(agent_id) if isinstance(llm, dict) else None
            if isinstance(entries, list):
                llm_calls_total = 0
                llm_send_message_count = 0
                for e in entries:
                    if isinstance(e, dict):
                        llm_calls_total += 1
                        plan = str(e.get("plan", "") or "")
                        if "[send_message]" in plan:
                            llm_send_message_count += 1
                    txt = _extract_output_text(e)
                    if not txt:
                        continue
                    n, tok = _num_tokens(txt)
                    toks.append(n)
                    tokenizer = tok
                if llm_calls_total > 0:
                    send_message_ratio_llm = llm_send_message_count / float(llm_calls_total)
            if isinstance(log.get("action"), dict):
                actions = log["action"].get(agent_id)
                if isinstance(actions, list) and actions:
                    env_total = 0
                    env_msg = 0
                    for a in actions:
                        if isinstance(a, str):
                            env_total += 1
                            if "[send_message]" in a:
                                env_msg += 1
                    if env_total > 0:
                        send_message_ratio_env = env_msg / float(env_total)
                        env_steps = env_total

    if not toks:
        actions_json = None
        if isinstance(objective, dict):
            actions_json = objective.get("actions_json")
        if isinstance(actions_json, str) and actions_json:
            p = Path(actions_json)
            if not p.exists():
                actions_folder = objective.get("actions_folder") if isinstance(objective, dict) else None
                if isinstance(actions_folder, str) and actions_folder:
                    global _PROAGENT_ACTIONS_INDEX
                    if _PROAGENT_ACTIONS_INDEX is None:
                        _PROAGENT_ACTIONS_INDEX = _build_proagent_actions_index()
                    hit = (_PROAGENT_ACTIONS_INDEX or {}).get(actions_folder)
                    if hit and hit.exists():
                        p = hit
        if str(p).endswith("actions.json") and p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                raw = None
            if isinstance(raw, dict):
                entries = raw.get(str(int(agent_id)))
                if isinstance(entries, list):
                    total_entries = 0
                    msg_entries = 0
                    max_t = -1
                    for e in entries:
                        if isinstance(e, dict):
                            total_entries += 1
                            m = e.get("message")
                            if isinstance(m, str) and m.strip():
                                msg_entries += 1
                            t = e.get("timestep")
                            try:
                                ti = int(t)
                            except Exception:
                                ti = None
                            if ti is not None and ti > max_t:
                                max_t = ti
                        txt = _extract_output_text(e)
                        if not txt:
                            continue
                        n, tok = _num_tokens(txt)
                        toks.append(n)
                        tokenizer = tok
                    if total_entries > 0:
                        send_message_ratio_llm = msg_entries / float(total_entries)
                        llm_calls_total = total_entries
                        llm_send_message_count = msg_entries
                    if max_t >= 0:
                        env_steps = max_t + 1
                        if total_entries > 0:
                            send_message_ratio_env = msg_entries / float(env_steps) if env_steps > 0 else None

    def _std_pop(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    if not toks:
        return {
            "tokenizer": tokenizer,
            "n_calls_with_text": 0,
            "tokens_sum": None,
            "tokens_mean": None,
            "tokens_std": None,
            "send_message_ratio_llm": send_message_ratio_llm,
            "send_message_ratio_env": send_message_ratio_env,
            "env_steps": env_steps,
            "llm_calls_total": llm_calls_total,
            "llm_send_message_count": llm_send_message_count,
        }

    vals = [float(x) for x in toks]
    return {
        "tokenizer": tokenizer,
        "n_calls_with_text": len(toks),
        "tokens_sum": int(sum(toks)),
        "tokens_mean": sum(vals) / len(vals),
        "tokens_std": _std_pop(vals),
        "send_message_ratio_llm": send_message_ratio_llm,
        "send_message_ratio_env": send_message_ratio_env,
        "env_steps": env_steps,
        "llm_calls_total": llm_calls_total,
        "llm_send_message_count": llm_send_message_count,
    }


def _msg_ratio_from_objective(obj: Any) -> Optional[float]:
    if not isinstance(obj, dict):
        return None
    try:
        send_steps = float(obj.get("agent_send_message_steps"))
        env_steps = float(obj.get("agent_env_steps"))
    except Exception:
        return None
    if env_steps <= 0:
        return None
    r = send_steps / env_steps
    if math.isnan(r) or math.isinf(r):
        return None
    return max(0.0, min(1.0, r))


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

    kk = float(k)
    x = (threshold - r) / threshold
    if kk <= 0:
        return max(0.0, min(max_penalty, max_penalty * x))

    denom = math.exp(kk) - 1.0
    if denom <= 0:
        return 0.0
    val = (math.exp(kk * x) - 1.0) / denom
    return max(0.0, min(max_penalty, max_penalty * val))


def _variant_from_log_path(log_path: str) -> Optional[str]:
    if not log_path:
        return None
    s = str(log_path)
    m = re.search(r"/(cwah-[^/]+)/runs/", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(cwah-[^/]+)\b", s)
    return m.group(1) if m else None


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


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
            val = _maybe_float(row.get("agent_llm_send_message_ratio_global"))
            if variant and val is not None:
                out[variant] = float(val)
    return out


def cmd_aggregate_results(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate window-level results to trajectory-level scores per dimension.")
    ap.add_argument("--results", required=True, help="Input results JSONL (from score-prompts).")
    ap.add_argument("--out", required=True, help="Output trajectory JSONL (one line per trajectory).")
    ap.add_argument("--out-csv", default=None, help="Optional CSV output (one row per trajectory).")
    ap.add_argument("--a", type=float, required=True, help="Weight a: penalty on (total violation count)/n.")
    ap.add_argument(
        "--b",
        type=float,
        required=True,
        help="Weight b: penalty on worst window score gap (max(5 - window_score)).",
    )
    ap.add_argument("--a-helpfulness", type=float, default=None, help="Override a for helpfulness (default: --a).")
    ap.add_argument("--b-helpfulness", type=float, default=None, help="Override b for helpfulness (default: --b).")
    ap.add_argument("--a-trustfulness", type=float, default=None, help="Override a for trustfulness (default: --a).")
    ap.add_argument("--b-trustfulness", type=float, default=None, help="Override b for trustfulness (default: --b).")
    ap.add_argument("--a-empathy", type=float, default=None, help="Override a for empathy (default: --a).")
    ap.add_argument("--b-empathy", type=float, default=None, help="Override b for empathy (default: --b).")
    ap.add_argument("--topk", type=int, default=10, help="Top-K violation rules to keep per dimension.")
    ap.add_argument(
        "--msg-penalty-threshold",
        type=float,
        default=0.15,
        help="If agent send_message ratio < threshold, apply a trajectory-level penalty to all dimensions.",
    )
    ap.add_argument(
        "--msg-penalty-max",
        type=float,
        default=1.0,
        help="Max penalty deducted from each dimension score (set 0 to disable).",
    )
    ap.add_argument(
        "--msg-penalty-k",
        type=float,
        default=4.0,
        help="Exponent shape parameter k (>0 steeper; <=0 falls back to linear).",
    )
    ap.add_argument(
        "--global-send-message-csv",
        default=None,
        help="If set, uses agent_llm_send_message_ratio_global (matched by variant extracted from log_path).",
    )

    args = ap.parse_args(argv)
    variant_llm_ratio_global = _load_variant_global_llm_send_message_ratio(args.global_send_message_csv)
    results_path = Path(args.results).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.out_csv).expanduser().resolve() if args.out_csv else None
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
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
            ],
        )
        csv_writer.writeheader()

    groups: Dict[Tuple[str, int, int, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for rec in _iter_jsonl(results_path):
        log_path = str(rec.get("log_path", ""))
        agent_id = _safe_int(rec.get("agent_id"), -1)
        human_id = _safe_int(rec.get("human_id"), -1)
        key = (
            log_path,
            agent_id,
            human_id,
            rec.get("task_id"),
            rec.get("env_id"),
            rec.get("task_name"),
        )
        groups[key].append(rec)

    def _objective_from_recs(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        for r in recs:
            obj = r.get("objective")
            if isinstance(obj, dict) and obj:
                return obj
        return {}

    def _ab_for_dim(dim: str) -> Tuple[float, float]:
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

    with out_path.open("w", encoding="utf-8") as out_f:
        for (log_path, agent_id, human_id, task_id, env_id, task_name), recs in groups.items():
            per_dim: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
            for r in recs:
                dim = str(r.get("dimension", "")).strip()
                if not dim:
                    continue
                window_id = str(r.get("window_id", "")).strip() or str(r.get("request_id", "")).strip()
                res = r.get("result", {})
                if not isinstance(res, dict):
                    continue
                per_dim[dim][window_id] = res

            n_windows = 0
            for dim, m in per_dim.items():
                n_windows = max(n_windows, len(m))

            traj: Dict[str, Any] = {
                "log_path": log_path,
                "agent_id": agent_id,
                "human_id": human_id,
                "task_id": task_id,
                "env_id": env_id,
                "task_name": task_name,
                "n_windows": n_windows,
                "weights": {
                    "a": args.a,
                    "b": args.b,
                    "helpfulness": {"a": args.a_helpfulness, "b": args.b_helpfulness},
                    "trustfulness": {"a": args.a_trustfulness, "b": args.b_trustfulness},
                    "empathy": {"a": args.a_empathy, "b": args.b_empathy},
                },
                "objective": _objective_from_recs(recs),
                "agent_send_message_ratio": None,
                "msg_penalty": None,
                "model_output_tokens": None,
                "dimensions": {},
            }

            mot = _output_token_stats_for_trajectory(
                log_path=log_path, objective=traj.get("objective"), agent_id=agent_id
            )
            traj["model_output_tokens"] = mot

            msg_ratio_env_steps = _msg_ratio_from_objective(traj.get("objective"))
            msg_ratio_llm = None
            if isinstance(mot, dict):
                try:
                    msg_ratio_llm = float(mot.get("send_message_ratio_llm"))
                except Exception:
                    msg_ratio_llm = None
                if msg_ratio_env_steps is None:
                    try:
                        msg_ratio_env_steps = float(mot.get("send_message_ratio_env"))
                    except Exception:
                        msg_ratio_env_steps = None

            msg_ratio_llm_global = None
            if variant_llm_ratio_global:
                variant = _variant_from_log_path(str(log_path))
                if variant:
                    msg_ratio_llm_global = variant_llm_ratio_global.get(variant)

            msg_ratio = msg_ratio_llm_global if msg_ratio_llm_global is not None else msg_ratio_env_steps
            msg_penalty = _msg_penalty_from_ratio(
                msg_ratio,
                threshold=float(args.msg_penalty_threshold),
                k=float(args.msg_penalty_k),
                max_penalty=float(args.msg_penalty_max),
            )
            traj["agent_send_message_ratio"] = msg_ratio
            traj["msg_penalty"] = msg_penalty
            traj["agent_send_message_ratio_llm"] = msg_ratio_llm
            traj["agent_send_message_ratio_env"] = msg_ratio
            traj["agent_send_message_ratio_env_steps"] = msg_ratio_env_steps
            traj["agent_send_message_ratio_llm_global"] = msg_ratio_llm_global

            if isinstance(traj.get("objective"), dict) and isinstance(mot, dict):
                if traj["objective"].get("agent_env_steps") in (None, "") and mot.get("env_steps") is not None:
                    traj["objective"]["agent_env_steps"] = mot.get("env_steps")

            for dim in ["helpfulness", "trustfulness", "empathy"]:
                windows_map = per_dim.get(dim, {})
                window_scores: List[float] = []
                window_viols: List[Any] = []
                for _, res in windows_map.items():
                    score = _safe_float(res.get("score"), 0.0)
                    violations = res.get("violations", [])
                    window_scores.append(score)
                    window_viols.append(violations)

                n = len(windows_map)
                viol_count_sum = sum(_violation_count(v) for v in window_viols)
                max_window_score_gap = max((5.0 - s) for s in window_scores) if window_scores else 0.0

                if n <= 0:
                    final_score = None
                else:
                    a, b = _ab_for_dim(dim)
                    final_score = 5.0 - a * (viol_count_sum / float(n)) - b * max_window_score_gap
                    if msg_penalty:
                        final_score = float(final_score) - float(msg_penalty)
                    final_score = max(0.0, min(5.0, final_score))

                traj["dimensions"][dim] = {
                    "score": final_score,
                    "n": n,
                    "violation_count_sum": viol_count_sum,
                    "max_window_score_gap": max_window_score_gap,
                    "mean_window_score": (sum(window_scores) / float(len(window_scores))) if window_scores else None,
                    "top_rules": _top_rules(window_viols, args.topk),
                }

            out_f.write(json.dumps(traj, ensure_ascii=False) + "\n")

            if csv_writer:
                mt = traj.get("model_output_tokens") if isinstance(traj.get("model_output_tokens"), dict) else {}
                csv_writer.writerow(
                    {
                        "log_path": log_path,
                        "agent_id": agent_id,
                        "human_id": human_id,
                        "task_id": task_id,
                        "env_id": env_id,
                        "task_name": task_name,
                        "n_windows": n_windows,
                        "helpfulness_score": traj["dimensions"]["helpfulness"]["score"],
                        "trustfulness_score": traj["dimensions"]["trustfulness"]["score"],
                        "empathy_score": traj["dimensions"]["empathy"]["score"],
                        "helpfulness_violation_count_sum": traj["dimensions"]["helpfulness"]["violation_count_sum"],
                        "trustfulness_violation_count_sum": traj["dimensions"]["trustfulness"]["violation_count_sum"],
                        "empathy_violation_count_sum": traj["dimensions"]["empathy"]["violation_count_sum"],
                        "helpfulness_max_window_score_gap": traj["dimensions"]["helpfulness"]["max_window_score_gap"],
                        "trustfulness_max_window_score_gap": traj["dimensions"]["trustfulness"]["max_window_score_gap"],
                        "empathy_max_window_score_gap": traj["dimensions"]["empathy"]["max_window_score_gap"],
                        "agent_send_message_ratio": msg_ratio,
                        "agent_send_message_ratio_env": msg_ratio,
                        "agent_send_message_ratio_env_steps": msg_ratio_env_steps,
                        "agent_send_message_ratio_llm": msg_ratio_llm,
                        "agent_send_message_ratio_llm_global": msg_ratio_llm_global,
                        "msg_penalty": msg_penalty,
                        "model_output_tokens_mean": mt.get("tokens_mean"),
                        "model_output_tokens_std": mt.get("tokens_std"),
                        "model_output_tokens_sum": mt.get("tokens_sum"),
                        "model_output_tokens_n_calls": mt.get("n_calls_with_text"),
                        "model_output_tokenizer": mt.get("tokenizer"),
                    }
                )

    if csv_file:
        csv_file.close()
    return 0


# =========================
# model-summary
# =========================

def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    return {"n": n, "mean": mean, "std": std, "min": min(values), "max": max(values)}


def cmd_model_summary(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute model-level mean scores across trajectories.")
    ap.add_argument("--trajectory", required=True, help="trajectory.jsonl from aggregate-results")
    ap.add_argument("--out-json", required=True, help="Output summary JSON")
    ap.add_argument("--out-csv", required=True, help="Output summary CSV")
    args = ap.parse_args(argv)

    traj_path = Path(args.trajectory).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dims = ["helpfulness", "trustfulness", "empathy"]
    dim_scores: Dict[str, List[float]] = {d: [] for d in dims}
    per_traj_overall: List[float] = []

    obj_total_steps: List[float] = []
    obj_agent_steps: List[float] = []
    obj_human_steps: List[float] = []
    obj_finished: List[float] = []

    out_tokens_mean: List[float] = []
    out_tokens_std: List[float] = []
    out_tokens_n_calls: List[float] = []

    n_traj = 0
    for rec in _iter_jsonl(traj_path):
        n_traj += 1
        drec = rec.get("dimensions", {})
        if not isinstance(drec, dict):
            continue
        obj = rec.get("objective", {})
        if isinstance(obj, dict):
            v = _safe_float_opt(obj.get("total_env_steps"))
            if v is not None:
                obj_total_steps.append(v)
            v = _safe_float_opt(obj.get("agent_env_steps"))
            if v is not None:
                obj_agent_steps.append(v)
            v = _safe_float_opt(obj.get("human_env_steps"))
            if v is not None:
                obj_human_steps.append(v)
            fin = obj.get("finished")
            if isinstance(fin, bool):
                obj_finished.append(1.0 if fin else 0.0)

        tok = rec.get("model_output_tokens")
        if isinstance(tok, dict):
            v = _safe_float_opt(tok.get("tokens_mean"))
            if v is not None:
                out_tokens_mean.append(v)
            v = _safe_float_opt(tok.get("tokens_std"))
            if v is not None:
                out_tokens_std.append(v)
            v = _safe_float_opt(tok.get("n_calls_with_text"))
            if v is not None:
                out_tokens_n_calls.append(v)

        scores = []
        for d in dims:
            s = _safe_float_opt((drec.get(d) or {}).get("score") if isinstance(drec.get(d), dict) else None)
            if s is not None:
                dim_scores[d].append(s)
                scores.append(s)
        if scores:
            per_traj_overall.append(sum(scores) / len(scores))

    summary = {
        "n_trajectories": n_traj,
        "objective": {
            "total_env_steps": _stats(obj_total_steps),
            "agent_env_steps": _stats(obj_agent_steps),
            "human_env_steps": _stats(obj_human_steps),
            "finished_rate": _stats(obj_finished),
        },
        "model_output_tokens": {
            "tokens_mean": _stats(out_tokens_mean),
            "tokens_std": _stats(out_tokens_std),
            "n_calls_with_text": _stats(out_tokens_n_calls),
        },
        "dimensions": {d: _stats(dim_scores[d]) for d in dims},
        "overall_per_trajectory_mean": _stats(per_traj_overall),
        "overall_of_dimension_means": None,
    }
    means = [summary["dimensions"][d]["mean"] for d in dims if summary["dimensions"][d]["mean"] is not None]
    if means:
        summary["overall_of_dimension_means"] = sum(means) / len(means)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n_trajectories",
                "total_env_steps_mean",
                "total_env_steps_std",
                "finished_rate_mean",
                "model_output_tokens_mean_mean",
                "model_output_tokens_mean_std",
                "model_output_tokens_std_mean",
                "model_output_tokens_std_std",
                "model_output_tokens_n_calls_mean",
                "model_output_tokens_n_calls_std",
                "helpfulness_mean",
                "trustfulness_mean",
                "empathy_mean",
                "overall_of_dimension_means",
                "overall_per_trajectory_mean",
                "overall_per_trajectory_std",
                "overall_per_trajectory_n",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "n_trajectories": n_traj,
                "total_env_steps_mean": summary["objective"]["total_env_steps"]["mean"],
                "total_env_steps_std": summary["objective"]["total_env_steps"]["std"],
                "finished_rate_mean": summary["objective"]["finished_rate"]["mean"],
                "model_output_tokens_mean_mean": summary["model_output_tokens"]["tokens_mean"]["mean"],
                "model_output_tokens_mean_std": summary["model_output_tokens"]["tokens_mean"]["std"],
                "model_output_tokens_std_mean": summary["model_output_tokens"]["tokens_std"]["mean"],
                "model_output_tokens_std_std": summary["model_output_tokens"]["tokens_std"]["std"],
                "model_output_tokens_n_calls_mean": summary["model_output_tokens"]["n_calls_with_text"]["mean"],
                "model_output_tokens_n_calls_std": summary["model_output_tokens"]["n_calls_with_text"]["std"],
                "helpfulness_mean": summary["dimensions"]["helpfulness"]["mean"],
                "trustfulness_mean": summary["dimensions"]["trustfulness"]["mean"],
                "empathy_mean": summary["dimensions"]["empathy"]["mean"],
                "overall_of_dimension_means": summary["overall_of_dimension_means"],
                "overall_per_trajectory_mean": summary["overall_per_trajectory_mean"]["mean"],
                "overall_per_trajectory_std": summary["overall_per_trajectory_mean"]["std"],
                "overall_per_trajectory_n": summary["overall_per_trajectory_mean"]["n"],
            }
        )

    return 0


# =========================
# compare-model-summaries
# =========================

def _find_outdirs(base: Path, pattern: str) -> List[Path]:
    return sorted([p for p in base.glob(pattern) if p.is_dir()])


def _load_summary(outdir: Path) -> Tuple[Dict[str, Any], str]:
    js = outdir / "model_summary.json"
    if js.exists():
        s = _read_json(js)
        dims = s.get("dimensions") or {}
        obj = s.get("objective") or {}
        overall_traj = s.get("overall_per_trajectory_mean") or {}
        row = {
            "name": outdir.name,
            "outdir": str(outdir),
            "n_trajectories": s.get("n_trajectories"),
            "total_env_steps_mean": (obj.get("total_env_steps") or {}).get("mean"),
            "total_env_steps_std": (obj.get("total_env_steps") or {}).get("std"),
            "finished_rate_mean": (obj.get("finished_rate") or {}).get("mean"),
            "helpfulness_mean": (dims.get("helpfulness") or {}).get("mean"),
            "trustfulness_mean": (dims.get("trustfulness") or {}).get("mean"),
            "empathy_mean": (dims.get("empathy") or {}).get("mean"),
            "overall_of_dimension_means": s.get("overall_of_dimension_means"),
            "overall_per_trajectory_mean": overall_traj.get("mean"),
            "overall_per_trajectory_std": overall_traj.get("std"),
            "overall_per_trajectory_n": overall_traj.get("n"),
        }
        return row, "json"

    cs = outdir / "model_summary.csv"
    if cs.exists():
        with cs.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            first = next(r, None)
        if not first:
            raise ValueError(f"Empty CSV: {cs}")
        row = {
            "name": outdir.name,
            "outdir": str(outdir),
            "n_trajectories": _safe_float_opt(first.get("n_trajectories")),
            "total_env_steps_mean": _safe_float_opt(first.get("total_env_steps_mean")),
            "total_env_steps_std": _safe_float_opt(first.get("total_env_steps_std")),
            "finished_rate_mean": _safe_float_opt(first.get("finished_rate_mean")),
            "helpfulness_mean": _safe_float_opt(first.get("helpfulness_mean")),
            "trustfulness_mean": _safe_float_opt(first.get("trustfulness_mean")),
            "empathy_mean": _safe_float_opt(first.get("empathy_mean")),
            "overall_of_dimension_means": _safe_float_opt(first.get("overall_of_dimension_means")),
            "overall_per_trajectory_mean": _safe_float_opt(first.get("overall_per_trajectory_mean")),
            "overall_per_trajectory_std": _safe_float_opt(first.get("overall_per_trajectory_std")),
            "overall_per_trajectory_n": _safe_float_opt(first.get("overall_per_trajectory_n")),
        }
        return row, "csv"

    raise FileNotFoundError(f"Missing model_summary.json/csv in {outdir}")


def _iter_targets(args: argparse.Namespace) -> List[Path]:
    if args.outdirs:
        outs: List[Path] = []
        for s in args.outdirs:
            p = Path(s).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            outs.append(p)
        return outs

    base = Path(args.base).expanduser()
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    return _find_outdirs(base, args.glob)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def cmd_compare_model_summaries(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate multiple completed out_* folders' model_summary into one comparison CSV."
    )
    ap.add_argument(
        "--base",
        default="coela_11/CoELA/evaluation",
        help="Directory to search when --outdirs is not provided (default: coela_11/CoELA/evaluation).",
    )
    ap.add_argument(
        "--glob",
        default="out_*",
        help="Glob pattern under --base for output folders (default: out_*).",
    )
    ap.add_argument(
        "--outdirs",
        nargs="*",
        default=None,
        help="Explicit output directories to include (overrides --base/--glob).",
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        help="Output comparison CSV path.",
    )
    ap.add_argument(
        "--sort-by",
        default="overall_of_dimension_means",
        help="Sort key (default: overall_of_dimension_means). Use empty string to disable sorting.",
    )
    ap.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending (default: ascending).",
    )
    args = ap.parse_args(argv)

    targets = _iter_targets(args)
    if not targets:
        raise SystemExit("No outdirs found.")

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for outdir in targets:
        try:
            row, src = _load_summary(outdir)
            row["source"] = src
            rows.append(row)
        except Exception as e:
            errors.append(f"{outdir}: {e}")

    if not rows:
        raise SystemExit("No valid summaries found.\n" + "\n".join(errors))

    fieldnames = [
        "name",
        "outdir",
        "source",
        "n_trajectories",
        "finished_rate_mean",
        "total_env_steps_mean",
        "total_env_steps_std",
        "helpfulness_mean",
        "trustfulness_mean",
        "empathy_mean",
        "overall_of_dimension_means",
        "overall_per_trajectory_mean",
        "overall_per_trajectory_std",
        "overall_per_trajectory_n",
    ]

    if args.sort_by:
        key = args.sort_by

        def sort_key(r: Dict[str, Any]) -> Tuple[int, Any]:
            v = r.get(key)
            return (1, 0) if v is None else (0, v)

        rows.sort(key=sort_key, reverse=args.descending)

    out_csv = Path(args.out_csv).expanduser()
    if not out_csv.is_absolute():
        out_csv = (Path.cwd() / out_csv).resolve()
    _write_csv(out_csv, rows, fieldnames)

    if errors:
        print("Warnings (skipped):")
        for msg in errors:
            print("  -", msg)

    print(f"Wrote: {out_csv} ({len(rows)} rows)")
    return 0


# =========================
# run-pipeline
# =========================

def cmd_run_pipeline(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run full llmjudge pipeline and produce process + detailed + compact outputs.")
    ap.add_argument("--runs", default=None, help="Runs directory (contains many run folders).")
    ap.add_argument(
        "--variant",
        default=None,
        help="Variant directory (e.g. coela_11/CoELA/cwah-0-agent-1-human_deepseekduida). Uses `<variant>/runs`.",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: evaluation/out_<variant_name> (derived from --variant or --runs).",
    )
    ap.add_argument("--agent-id", type=int, required=True)
    ap.add_argument("--human-id", type=int, required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--a-helpfulness", type=float, default=None)
    ap.add_argument("--b-helpfulness", type=float, default=None)
    ap.add_argument("--a-trustfulness", type=float, default=None)
    ap.add_argument("--b-trustfulness", type=float, default=None)
    ap.add_argument("--a-empathy", type=float, default=None)
    ap.add_argument("--b-empathy", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-proxy-host", default="localhost")

    args = ap.parse_args(argv)
    if bool(args.runs) == bool(args.variant):
        raise SystemExit("Provide exactly one of: --runs or --variant")

    base = Path(__file__).resolve().parent
    coela_root = base.parent

    if args.variant:
        variant_dir = Path(args.variant).expanduser()
        if not variant_dir.is_absolute():
            variant_dir = (coela_root / variant_dir).resolve()
        runs_dir = variant_dir / "runs"
        if args.outdir:
            outdir = Path(args.outdir).expanduser().resolve()
        else:
            outdir = (coela_root / "evaluation" / f"out_{variant_dir.name}").resolve()
    else:
        runs_dir = Path(args.runs).expanduser()
        if not runs_dir.is_absolute():
            runs_dir = (coela_root / runs_dir).resolve()
        if args.outdir:
            outdir = Path(args.outdir).expanduser().resolve()
        else:
            base_name = runs_dir.parent.name if runs_dir.name == "runs" else runs_dir.name
            outdir = (coela_root / "evaluation" / f"out_{base_name}").resolve()

    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    outdir.mkdir(parents=True, exist_ok=True)

    windows_jsonl = str(outdir / "windows.jsonl")
    windows_preview = str(outdir / "windows_preview.txt")
    requests_jsonl = str(outdir / "requests_thin.jsonl")
    results_lite = str(outdir / "results_lite.jsonl")
    results_lite_obj = str(outdir / "results_lite_obj.jsonl")
    detailed = str(outdir / "results_detailed.jsonl")
    compact = str(outdir / "results_compact.jsonl")
    window_csv = str(outdir / "window_scores.csv")
    traj_jsonl = str(outdir / "trajectory.jsonl")
    traj_csv = str(outdir / "trajectory.csv")
    model_summary_json = str(outdir / "model_summary.json")
    model_summary_csv = str(outdir / "model_summary.csv")

    os.environ.setdefault("NO_PROXY", args.no_proxy_host)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("http_proxy", None)

    cmd_build_windows(
        [
            "--input",
            str(runs_dir),
            "--out",
            windows_jsonl,
            "--preview-out",
            windows_preview,
            "--agent-id",
            str(args.agent_id),
            "--human-id",
            str(args.human_id),
            "--max-chars",
            str(args.max_chars),
        ]
        + (["--max-files", str(args.max_files)] if args.max_files is not None else [])
    )

    cmd_build_prompts(["--windows", windows_jsonl, "--out", requests_jsonl, "--thin"])

    score_cmd = [
        "--prompts",
        requests_jsonl,
        "--windows",
        windows_jsonl,
        "--out",
        results_lite,
        "--api-base",
        args.api_base,
        "--model",
        args.model,
        "--api-key",
        args.api_key,
        "--workers",
        str(args.workers),
    ]
    if args.resume:
        score_cmd.append("--resume")
    cmd_score_prompts(score_cmd)

    cmd_backfill_objective(
        [
            "--windows",
            windows_jsonl,
            "--results",
            results_lite,
            "--out",
            results_lite_obj,
            "--overwrite",
        ]
    )

    cmd_merge_outputs(
        [
            "--windows",
            windows_jsonl,
            "--results",
            results_lite_obj,
            "--out-detailed",
            detailed,
            "--out-compact",
            compact,
        ]
    )

    cmd_export_window_csv(["--results", results_lite_obj, "--out-csv", window_csv])

    agg_cmd = [
        "--results",
        results_lite_obj,
        "--out",
        traj_jsonl,
        "--out-csv",
        traj_csv,
        "--a",
        str(args.a),
        "--b",
        str(args.b),
    ]
    for flag, val in [
        ("--a-helpfulness", args.a_helpfulness),
        ("--b-helpfulness", args.b_helpfulness),
        ("--a-trustfulness", args.a_trustfulness),
        ("--b-trustfulness", args.b_trustfulness),
        ("--a-empathy", args.a_empathy),
        ("--b-empathy", args.b_empathy),
    ]:
        if val is not None:
            agg_cmd += [flag, str(val)]
    cmd_aggregate_results(agg_cmd)

    cmd_model_summary(
        [
            "--trajectory",
            traj_jsonl,
            "--out-json",
            model_summary_json,
            "--out-csv",
            model_summary_csv,
        ]
    )

    return 0


# =========================
# main dispatch
# =========================

def _print_main_help() -> None:
    cmds = [
        "build-windows",
        "build-prompts",
        "score-prompts",
        "backfill-objective",
        "merge-outputs",
        "export-window-csv",
        "aggregate-results",
        "model-summary",
        "compare-model-summaries",
        "run-pipeline",
    ]
    print("Usage: python evaluation/llmjudge.py <command> [args]")
    print("Commands:")
    for c in cmds:
        print(f"  - {c}")
    print("Run: python evaluation/llmjudge.py <command> --help")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_main_help()
        return 0

    cmd = argv[0]
    args = argv[1:]
    commands = {
        "build-windows": cmd_build_windows,
        "build-prompts": cmd_build_prompts,
        "score-prompts": cmd_score_prompts,
        "backfill-objective": cmd_backfill_objective,
        "merge-outputs": cmd_merge_outputs,
        "export-window-csv": cmd_export_window_csv,
        "aggregate-results": cmd_aggregate_results,
        "model-summary": cmd_model_summary,
        "compare-model-summaries": cmd_compare_model_summaries,
        "run-pipeline": cmd_run_pipeline,
    }
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        _print_main_help()
        return 2

    return commands[cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
