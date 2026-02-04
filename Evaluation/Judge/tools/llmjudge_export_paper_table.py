#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_TIKTOKEN_ENCODING = None
_PROAGENT_ACTIONS_INDEX: Optional[Dict[str, Path]] = None


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    import tiktoken

    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None or getattr(_TIKTOKEN_ENCODING, "name", None) != encoding_name:
        _TIKTOKEN_ENCODING = tiktoken.get_encoding(encoding_name)
    return len(_TIKTOKEN_ENCODING.encode(string, disallowed_special=()))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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


def _std_population(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def _load_model_summary(outdir: Path) -> Dict[str, Any]:
    p = outdir / "model_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing model_summary.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_outdir_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (role, model_key)
      role: "agent1" corresponds to evaluated agent_id=0; "agent2" corresponds to evaluated agent_id=1.
    """
    m = re.match(r"^out_cwah-0-agent-1-human_(.+)$", name)
    if m:
        return "agent1", m.group(1)
    m = re.match(r"^out_cwah-0-human-1-agent_(.+)$", name)
    if m:
        return "agent2", m.group(1)

    # ProAgent outputs: out_src-<agent>-agent-<human>-human-... OR out_src-<human>-human-<agent>-agent...
    m = re.match(r"^out_src-(\d+)-agent-(\d+)-human[_-](.+)$", name)
    if m:
        agent_id = int(m.group(1))
        role = "agent1" if agent_id == 0 else ("agent2" if agent_id == 1 else None)
        mk = m.group(3)
        mk = re.sub(r"^\d+-", "", mk)  # strip leading run tag like "1229-"
        return role, mk
    m = re.match(r"^out_src-(\d+)-human-(\d+)-agent[_-](.+)$", name)
    if m:
        agent_id = int(m.group(2))
        role = "agent1" if agent_id == 0 else ("agent2" if agent_id == 1 else None)
        mk = m.group(3)
        mk = re.sub(r"^\d+-", "", mk)
        return role, mk
    return None, None


def _canonical_model_key(model_key: str) -> str:
    # Merge split runs like `2deepseek-part1` / `2deepseek-part2` into `2deepseek`.
    mk = re.sub(r"([_-])part\d+$", "", model_key, flags=re.IGNORECASE)
    mkl = mk.lower()
    # Normalize common aliases so Agent1/Agent2 land in the same row.
    if mkl in ("qwen72b", "deepseek_qwen72b"):
        return "qwen72"
    if mkl in ("deepseek_qwen38b",):
        return "qwen38b"
    return mk


def _method_for_outdir(outdir_name: str, model_key: str) -> str:
    if outdir_name.startswith("out_src-"):
        return "ProAgent"
    mk = model_key.lower()
    return "SynerMate" if ("rl" in mk) else "CoELA"


def _llm_display_name(model_key: str) -> str:
    m = model_key.lower()
    # ProAgent labels: keep as-is (but try to map common ones).
    if m in ("2deepseek", "deepseek", "deepseek2"):
        return "DeepSeek-V3.1"
    if m in ("ours",):
        return "Ours"
    if m in ("qwen7b", "noqwen7b_deepseek", "qwen7b_deepseek"):
        return "Qwen2.5-7B-Instruct"
    if m in ("qwen8b",):
        return "Qwen2.5-8B"
    if m in ("qwen72", "qwen72b", "deepseek_qwen72b"):
        return "Qwen2.5-72B-Instruct"
    if m in ("deepseek_qwen38b",):
        return "Qwen2.5-3B"
    if m in ("qwen38b",):
        return "Qwen2.5-3B"

    if m == "deepseekduida":
        return "DeepSeek-V3.1"
    if m in ("qwen72b", "qwen72"):
        return "Qwen2.5-72B-Instruct"
    if m in ("qwen7b",):
        return "Qwen2.5-7B-Instruct"
    if m in ("qwen7brl",):
        return "Qwen2.5-7B-Instruct"
    if m in ("qwen3",):
        return "Qwen3"
    if m in ("qwen3rl",):
        return "Qwen3"
    return model_key


def _skip_model(model_key: str) -> bool:
    m = model_key.lower()
    return m.startswith("qwen3")


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


def _get_proagent_root() -> Optional[Path]:
    # Repo layout: <repo>/coela_11/CoELA/evaluation/this_file.py, and <repo>/ProAgent_1221 exists.
    repo_root = Path(__file__).resolve().parents[3]
    p = repo_root / "ProAgent_1221"
    return p if p.exists() else None


def _build_proagent_actions_index() -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    root = _get_proagent_root()
    if not root:
        return idx
    # Expected layout: ProAgent_1221/src-*/actions/<actions_folder>/step_1/actions.json
    for p in root.glob("src-*/actions/*/step_1/actions.json"):
        actions_folder = p.parent.parent.name
        if actions_folder not in idx:
            idx[actions_folder] = p
    return idx


def _resolve_actions_json_path(actions_json: str, objective: Dict[str, Any], *, outdir: Optional[Path] = None) -> Optional[Path]:
    p = Path(actions_json)
    if p.exists():
        return p

    # Common: user renamed / split data_root; try using outdir suffix like "-part1".
    data_root = objective.get("data_root")
    if isinstance(data_root, str) and outdir is not None:
        m = re.search(r"([_-]part\d+)$", outdir.name, flags=re.IGNORECASE)
        if m:
            part_suffix = m.group(1)
            if not data_root.lower().endswith(part_suffix.lower()):
                cand_root = data_root + part_suffix
                cand = Path(actions_json.replace(data_root, cand_root, 1))
                if cand.exists():
                    return cand

    # Fallback: resolve by actions_folder name via a cached index under ProAgent_1221.
    actions_folder = objective.get("actions_folder")
    if isinstance(actions_folder, str) and actions_folder:
        global _PROAGENT_ACTIONS_INDEX
        if _PROAGENT_ACTIONS_INDEX is None:
            _PROAGENT_ACTIONS_INDEX = _build_proagent_actions_index()
        hit = (_PROAGENT_ACTIONS_INDEX or {}).get(actions_folder)
        if hit and hit.exists():
            return hit
    return None


def _trajectory_token_total_k(traj_jsonl: Path) -> Optional[float]:
    """
    Mean output tokens per sampled step (LLM call), in thousands.
    Prefers cached `model_output_tokens.tokens_mean` in trajectory.jsonl (fast).
    Falls back to loading raw `.pik` / `actions.json` when missing (slow).
    """
    per_traj_means_k: List[float] = []
    for tr in _iter_jsonl(traj_jsonl):
        # Fast path: reuse cached tokens from llmjudge_aggregate_results.py
        mot = tr.get("model_output_tokens")
        if isinstance(mot, dict):
            tm = _safe_float(mot.get("tokens_mean"))
            if tm is not None:
                per_traj_means_k.append(float(tm) / 1000.0)
                continue

        log_path = tr.get("log_path")
        agent_id = tr.get("agent_id")
        if not isinstance(log_path, str):
            continue
        try:
            aid = int(agent_id)
        except Exception:
            continue

        # CWAH: pik logs
        if log_path.endswith(".pik"):
            try:
                log = pickle.load(open(log_path, "rb"))
            except Exception:
                continue
            llm = log.get("LLM", {})
            entries = llm.get(aid) if isinstance(llm, dict) else None
            if not isinstance(entries, list) or not entries:
                continue
            total = 0
            n_calls = 0
            for e in entries:
                txt = _extract_output_text(e)
                if txt:
                    total += num_tokens_from_string(txt)
                    n_calls += 1
            if n_calls > 0:
                per_traj_means_k.append((total / float(n_calls)) / 1000.0)
            continue

        # ProAgent: actions.json path stored in objective/actions_json
        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue
        actions_json = obj.get("actions_json") or log_path
        if not isinstance(actions_json, str) or not actions_json.endswith("actions.json"):
            continue
        p = _resolve_actions_json_path(actions_json, obj, outdir=traj_jsonl.parent)
        if not p:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        entries = raw.get(str(aid))
        if not isinstance(entries, list) or not entries:
            continue
        total = 0
        n_calls = 0
        for e in entries:
            txt = _extract_output_text(e)
            if txt:
                total += num_tokens_from_string(txt)
                n_calls += 1
        if n_calls > 0:
            per_traj_means_k.append((total / float(n_calls)) / 1000.0)
    return _mean(per_traj_means_k)


def _trajectory_tokens_per_traj_k(traj_jsonl: Path) -> List[float]:
    """
    Per-trajectory mean output tokens per sampled step (LLM call), in thousands.
    """
    per_traj_means_k: List[float] = []
    for tr in _iter_jsonl(traj_jsonl):
        # Fast path: reuse cached tokens from llmjudge_aggregate_results.py
        mot = tr.get("model_output_tokens")
        if isinstance(mot, dict):
            tm = _safe_float(mot.get("tokens_mean"))
            if tm is not None:
                per_traj_means_k.append(float(tm) / 1000.0)
                continue

        log_path = tr.get("log_path")
        agent_id = tr.get("agent_id")
        if not isinstance(log_path, str):
            continue
        try:
            aid = int(agent_id)
        except Exception:
            continue

        if log_path.endswith(".pik"):
            try:
                log = pickle.load(open(log_path, "rb"))
            except Exception:
                continue
            llm = log.get("LLM", {})
            entries = llm.get(aid) if isinstance(llm, dict) else None
            if not isinstance(entries, list) or not entries:
                continue
            total = 0
            n_calls = 0
            for e in entries:
                txt = _extract_output_text(e)
                if txt:
                    total += num_tokens_from_string(txt)
                    n_calls += 1
            if n_calls > 0:
                per_traj_means_k.append((total / float(n_calls)) / 1000.0)
            continue

        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue
        actions_json = obj.get("actions_json") or log_path
        if not isinstance(actions_json, str) or not actions_json.endswith("actions.json"):
            continue
        p = _resolve_actions_json_path(actions_json, obj, outdir=traj_jsonl.parent)
        if not p:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        entries = raw.get(str(aid))
        if not isinstance(entries, list) or not entries:
            continue
        total = 0
        n_calls = 0
        for e in entries:
            txt = _extract_output_text(e)
            if txt:
                total += num_tokens_from_string(txt)
                n_calls += 1
        if n_calls > 0:
            per_traj_means_k.append((total / float(n_calls)) / 1000.0)
    return per_traj_means_k


def _trajectory_env_steps_mean_std_from_actions(traj_jsonl: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    ProAgent: env steps are derived from actions.json entries' `timestep` (max+1).
    Returns (mean, std) across trajectories.
    """
    steps: List[float] = []
    for tr in _iter_jsonl(traj_jsonl):
        # Fast path: reuse cached objective agent_env_steps when present.
        obj0 = tr.get("objective")
        if isinstance(obj0, dict):
            v = _safe_float(obj0.get("agent_env_steps"))
            if v is not None:
                steps.append(v)
                continue

        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue
        actions_json = obj.get("actions_json")
        agent_id = obj.get("agent_id")
        if not isinstance(actions_json, str) or not actions_json:
            continue
        try:
            aid = int(agent_id)
        except Exception:
            continue
        p = _resolve_actions_json_path(actions_json, obj, outdir=traj_jsonl.parent)
        if not p:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        entries = raw.get(str(aid))
        if not isinstance(entries, list) or not entries:
            continue
        max_t = -1
        for e in entries:
            if not isinstance(e, dict):
                continue
            t = e.get("timestep")
            try:
                ti = int(t)
            except Exception:
                continue
            if ti > max_t:
                max_t = ti
        if max_t >= 0:
            steps.append(float(max_t + 1))
    return _mean(steps), _std_population(steps)


def _trajectory_env_steps_per_traj_from_actions(traj_jsonl: Path) -> List[float]:
    steps: List[float] = []
    for tr in _iter_jsonl(traj_jsonl):
        # Fast path: reuse cached objective agent_env_steps when present.
        obj0 = tr.get("objective")
        if isinstance(obj0, dict):
            v = _safe_float(obj0.get("agent_env_steps"))
            if v is not None:
                steps.append(v)
                continue

        obj = tr.get("objective")
        if not isinstance(obj, dict):
            continue
        actions_json = obj.get("actions_json")
        agent_id = obj.get("agent_id")
        if not isinstance(actions_json, str) or not actions_json:
            continue
        try:
            aid = int(agent_id)
        except Exception:
            continue
        p = _resolve_actions_json_path(actions_json, obj, outdir=traj_jsonl.parent)
        if not p:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        entries = raw.get(str(aid))
        if not isinstance(entries, list) or not entries:
            continue
        max_t = -1
        for e in entries:
            if not isinstance(e, dict):
                continue
            t = e.get("timestep")
            try:
                ti = int(t)
            except Exception:
                continue
            if ti > max_t:
                max_t = ti
        if max_t >= 0:
            steps.append(float(max_t + 1))
    return steps


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export paper-style comparison table for CWAH evaluation outdirs.")
    ap.add_argument("--base", default="coela_11/CoELA/evaluation", help="Evaluation directory containing out_* folders.")
    ap.add_argument("--glob", default="out_cwah-*", help="Glob pattern to scan under --base.")
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (backward compatible: writes normalized table). Prefer --out-csv-raw/--out-csv-norm.",
    )
    ap.add_argument("--out-csv-raw", default=None, help="Output CSV path for unnormalized (raw) affective means.")
    ap.add_argument("--out-csv-norm", default=None, help="Output CSV path for normalized (zscore->sigmoid) affective means.")
    args = ap.parse_args(argv)

    base = Path(args.base).expanduser().resolve()
    out_csv_raw = Path(args.out_csv_raw).expanduser().resolve() if args.out_csv_raw else None
    out_csv_norm = Path(args.out_csv_norm).expanduser().resolve() if args.out_csv_norm else None
    # Backward compatibility: --out-csv writes normalized table.
    if args.out_csv and not out_csv_norm and not out_csv_raw:
        out_csv_norm = Path(args.out_csv).expanduser().resolve()

    if not out_csv_raw and not out_csv_norm:
        raise SystemExit("Provide at least one of: --out-csv-raw, --out-csv-norm (or legacy --out-csv).")
    if out_csv_raw:
        out_csv_raw.parent.mkdir(parents=True, exist_ok=True)
    if out_csv_norm:
        out_csv_norm.parent.mkdir(parents=True, exist_ok=True)

    # Collect per model_key per role, merging multi-part runs by pooling trajectories/windows.
    per_model: Dict[str, Dict[str, Any]] = {}
    for outdir in sorted([p for p in base.glob(args.glob) if p.is_dir()]):
        role, model_key = _parse_outdir_name(outdir.name)
        if not role or not model_key:
            continue
        model_key = _canonical_model_key(model_key)
        # Only skip Qwen3 for CWAH tables (it won't appear in ProAgent tables anyway).
        if _skip_model(model_key) and outdir.name.startswith("out_cwah-"):
            continue
        try:
            ms = _load_model_summary(outdir)
        except Exception:
            continue
        dims = ms.get("dimensions") if isinstance(ms.get("dimensions"), dict) else {}
        obj = ms.get("objective") if isinstance(ms.get("objective"), dict) else {}
        steps_obj_total = obj.get("total_env_steps") if isinstance(obj.get("total_env_steps"), dict) else {}
        steps_obj_agent = obj.get("agent_env_steps") if isinstance(obj.get("agent_env_steps"), dict) else {}

        rec = per_model.setdefault(model_key, {"method": _method_for_outdir(outdir.name, model_key)})
        rec["method"] = rec.get("method") or _method_for_outdir(outdir.name, model_key)
        rrec = rec.setdefault(
            role,
            {
                "steps_list": [],
                "tokens_k_list": [],
                "help_sum": 0.0,
                "help_n": 0,
                "trust_sum": 0.0,
                "trust_n": 0,
                "empathy_sum": 0.0,
                "empathy_n": 0,
                "steps_mean_fallback": None,
                "steps_std_fallback": None,
            },
        )

        # Affective pooling via (mean * n) to merge split runs.
        for dim, sum_key, n_key in [
            ("helpfulness", "help_sum", "help_n"),
            ("trustfulness", "trust_sum", "trust_n"),
            ("empathy", "empathy_sum", "empathy_n"),
        ]:
            d = dims.get(dim) if isinstance(dims.get(dim), dict) else None
            if isinstance(d, dict):
                n = d.get("n")
                mean = d.get("mean")
                try:
                    ni = int(n)
                    mv = float(mean)
                except Exception:
                    continue
                if ni > 0:
                    rrec[sum_key] += mv * ni
                    rrec[n_key] += ni

        # Steps mean/std from model_summary.
        # - CWAH: total_env_steps mean/std
        # - ProAgent: agent_env_steps mean/std (cached into objective by llmjudge_aggregate_results.py)
        if outdir.name.startswith("out_cwah-") and isinstance(steps_obj_total, dict):
            if rrec.get("steps_mean_fallback") in (None, ""):
                rrec["steps_mean_fallback"] = steps_obj_total.get("mean")
                rrec["steps_std_fallback"] = steps_obj_total.get("std")
        if outdir.name.startswith("out_src-") and isinstance(steps_obj_agent, dict):
            if rrec.get("steps_mean_fallback") in (None, ""):
                rrec["steps_mean_fallback"] = steps_obj_agent.get("mean")
                rrec["steps_std_fallback"] = steps_obj_agent.get("std")

        # Tokens(k): prefer model_summary.model_output_tokens.tokens_mean.mean (tokens per call), else fall back to trajectory scanning.
        mot = ms.get("model_output_tokens") if isinstance(ms.get("model_output_tokens"), dict) else {}
        mot_mean = None
        if isinstance(mot, dict):
            mm = mot.get("tokens_mean")
            if isinstance(mm, dict):
                mot_mean = _safe_float(mm.get("mean"))
        if mot_mean is not None:
            rrec["tokens_k_list"].append(mot_mean / 1000.0)
        else:
            traj = outdir / "trajectory.jsonl"
            if traj.exists():
                rrec["tokens_k_list"].extend(_trajectory_tokens_per_traj_k(traj))
                if outdir.name.startswith("out_src-"):
                    rrec["steps_list"].extend(_trajectory_env_steps_per_traj_from_actions(traj))

    # Finalize pooled aggregates into flat fields (for easy CSV writing).
    for mk, rec in per_model.items():
        for role in ["agent1", "agent2"]:
            rrec = rec.get(role)
            if not isinstance(rrec, dict):
                continue
            steps_list = [x for x in rrec.get("steps_list", []) if isinstance(x, (int, float))]
            tokens_list = [x for x in rrec.get("tokens_k_list", []) if isinstance(x, (int, float))]
            steps_mean = _mean([float(x) for x in steps_list]) if steps_list else _safe_float(rrec.get("steps_mean_fallback"))
            steps_std = _std_population([float(x) for x in steps_list]) if steps_list else _safe_float(rrec.get("steps_std_fallback"))
            rec[f"{role}_steps_mean"] = steps_mean
            rec[f"{role}_steps_std"] = steps_std
            rec[f"{role}_tokens_k"] = _mean([float(x) for x in tokens_list]) if tokens_list else None

            for dim, sum_key, n_key in [
                ("helpfulness", "help_sum", "help_n"),
                ("trustfulness", "trust_sum", "trust_n"),
                ("empathy", "empathy_sum", "empathy_n"),
            ]:
                n = rrec.get(n_key) or 0
                s = rrec.get(sum_key) or 0.0
                rec[f"{role}_{dim}_mean"] = (s / n) if n else None

    # Zscore+sigmoid for affective means (per role, per dimension) across all models.
    for role in ["agent1", "agent2"]:
        for dim in ["helpfulness", "trustfulness", "empathy"]:
            vals: List[float] = []
            for mk, rec in per_model.items():
                v = _safe_float(rec.get(f"{role}_{dim}_mean"))
                if v is not None:
                    vals.append(v)
            m = _mean(vals) or 0.0
            s = _std_population(vals) or 0.0
            for mk, rec in per_model.items():
                v = _safe_float(rec.get(f"{role}_{dim}_mean"))
                if v is None or s == 0.0:
                    rec[f"{role}_{dim}_sigmoid"] = None
                else:
                    rec[f"{role}_{dim}_sigmoid"] = _sigmoid((v - m) / s)

    # Write CSV with multi-row header similar to the example.
    columns = [
        "Method",
        "LLMs",
        "Agent 1 Score",
        "Agent 2 Score",
        "Agent 1 Std.",
        "Agent 2 Std.",
        "Agent 1 #Tokens(k)",
        "Agent 2 #Tokens(k)",
        "Agent 1 Helpfulness",
        "Agent 2 Helpfulness",
        "Agent 1 Trustfulness",
        "Agent 2 Trustfulness",
        "Agent 1 Empathy",
        "Agent 2 Empathy",
    ]

    def fmt(x: Any, nd: int) -> Any:
        v = _safe_float(x)
        return (round(v, nd) if v is not None else "")

    def write_table(out_path: Path, *, normalized: bool) -> None:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "", "Efficiency", "", "", "", "", "", "Affective", "", "", "", "", ""])
            w.writerow(["", "", "Score", "", "Std.", "", "#Tokens(k)", "", "Helpfulness", "", "Trustfulness", "", "Empathy", ""])
            w.writerow(columns)

            for model_key in sorted(
                per_model.keys(), key=lambda x: (per_model.get(x, {}).get("method", ""), _llm_display_name(x), x)
            ):
                rec = per_model[model_key]
                row = [
                    rec.get("method") or "",
                    _llm_display_name(model_key),
                    rec.get("agent1_steps_mean"),
                    rec.get("agent2_steps_mean"),
                    rec.get("agent1_steps_std"),
                    rec.get("agent2_steps_std"),
                    rec.get("agent1_tokens_k"),
                    rec.get("agent2_tokens_k"),
                    rec.get("agent1_helpfulness_sigmoid" if normalized else "agent1_helpfulness_mean"),
                    rec.get("agent2_helpfulness_sigmoid" if normalized else "agent2_helpfulness_mean"),
                    rec.get("agent1_trustfulness_sigmoid" if normalized else "agent1_trustfulness_mean"),
                    rec.get("agent2_trustfulness_sigmoid" if normalized else "agent2_trustfulness_mean"),
                    rec.get("agent1_empathy_sigmoid" if normalized else "agent1_empathy_mean"),
                    rec.get("agent2_empathy_sigmoid" if normalized else "agent2_empathy_mean"),
                ]

                row_fmt = [
                    row[0],
                    row[1],
                    fmt(row[2], 2),
                    fmt(row[3], 2),
                    fmt(row[4], 2),
                    fmt(row[5], 2),
                    fmt(row[6], 2),
                    fmt(row[7], 2),
                    fmt(row[8], 3),
                    fmt(row[9], 3),
                    fmt(row[10], 3),
                    fmt(row[11], 3),
                    fmt(row[12], 3),
                    fmt(row[13], 3),
                ]
                w.writerow(row_fmt)

    if out_csv_raw:
        write_table(out_csv_raw, normalized=False)
        print(f"Wrote: {out_csv_raw}")
    if out_csv_norm:
        write_table(out_csv_norm, normalized=True)
        print(f"Wrote: {out_csv_norm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
