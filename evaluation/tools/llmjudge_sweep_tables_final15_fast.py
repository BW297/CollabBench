#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
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
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize_sigmoid_z(values_by_row: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [(k, v) for k, v in values_by_row.items() if v is not None]
    if not vals:
        return {k: None for k in values_by_row}
    xs = [float(v) for _, v in vals]
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var)
    out: Dict[str, Optional[float]] = {}
    for k, v in values_by_row.items():
        if v is None:
            out[k] = None
            continue
        if sd == 0:
            out[k] = _sigmoid(0.0)
        else:
            out[k] = _sigmoid((float(v) - mu) / sd)
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


def _parse_outdir_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"^out_cwah-0-agent-1-human_(.+)$", name)
    if m:
        return "agent1", m.group(1)
    m = re.match(r"^out_cwah-0-human-1-agent_(.+)$", name)
    if m:
        return "agent2", m.group(1)

    m = re.match(r"^out_src-(\d+)-agent-(\d+)-human[_-](.+)$", name)
    if m:
        agent_id = int(m.group(1))
        role = "agent1" if agent_id == 0 else ("agent2" if agent_id == 1 else None)
        return role, m.group(3)
    m = re.match(r"^out_src-(\d+)-human-(\d+)-agent[_-](.+)$", name)
    if m:
        agent_id = int(m.group(2))
        role = "agent1" if agent_id == 0 else ("agent2" if agent_id == 1 else None)
        return role, m.group(3)
    return None, None


def _canonical_model_key(model_key: str) -> str:
    mk = re.sub(r"([_-])part\d+$", "", model_key, flags=re.IGNORECASE)
    mkl = mk.lower()
    if mkl in ("2deepseek", "deepseek", "deepseek2", "deepseekduida"):
        return "deepseek"
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
    if m in ("2deepseek", "deepseek", "deepseek2", "deepseekduida"):
        return "DeepSeek-V3.1"
    if m in ("qwen72", "qwen72b", "deepseek_qwen72b"):
        return "Qwen2.5-72B-Instruct"
    if m in ("qwen7b", "qwen7brl"):
        return "Qwen2.5-7B-Instruct"
    if m in ("gpt",):
        return "GPT"
    if m in ("qwen3", "qwen3rl"):
        return "Qwen3"
    return model_key


def _split_list(s: str) -> List[str]:
    parts = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            parts.append(x)
    return parts


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in _split_list(s)]


# -------------------- Final-15 filters --------------------

COELA_FINAL15: Dict[str, List[int]] = {
    "0_read_book": [0, 7, 4, 1, 13, 18, 12, 5, 25, 2, 15, 29, 16, 14, 27],
    "5_read_book": [29, 17, 8, 4, 24, 7, 6, 18, 9, 1, 2, 23, 28, 14, 25],
    "10_put_dishwasher": [11, 3, 15, 13, 19, 18, 28, 14, 25, 29, 9, 10, 12, 6, 7],
    "16_put_dishwasher": [14, 17, 7, 29, 25, 26, 15, 20, 4, 22, 9, 24, 18, 2, 8],
    "20_prepare_food": [18, 2, 29, 3, 7, 27, 0, 11, 28, 9, 23, 19, 12, 5, 14],
    "26_prepare_food": [1, 22, 2, 8, 26, 14, 17, 10, 6, 21, 7, 4, 13, 29, 25],
    "30_put_fridge": [10, 5, 21, 7, 12, 15, 28, 4, 0, 24, 11, 20, 25, 18, 14],
    "32_put_fridge": [19, 15, 20, 1, 11, 18, 7, 28, 27, 6, 8, 26, 5, 0, 3],
    "40_setup_table": [8, 10, 2, 29, 17, 11, 7, 12, 20, 26, 22, 21, 5, 16, 14],
    "49_setup_table": [20, 29, 24, 4, 26, 15, 12, 22, 21, 8, 1, 19, 14, 0, 25],
}

PROAGENT_FINAL15: Dict[str, List[int]] = {
    "asymmetric_advantages": [2, 4, 5, 6, 8, 9, 10, 12, 18, 19, 20, 21, 22, 23, 28],
    "coordination_ring": [3, 4, 6, 11, 12, 13, 14, 19, 21, 23, 24, 26, 27, 28, 29],
    "counter_circuit": [0, 1, 2, 3, 6, 9, 10, 17, 20, 21, 22, 25, 26, 27, 28],
    "cramped_room": [0, 2, 5, 7, 8, 9, 10, 11, 12, 14, 19, 22, 23, 24, 28],
    "forced_coordination": [1, 3, 4, 5, 6, 8, 10, 12, 13, 15, 18, 20, 22, 26, 29],
}


def _coela_taskkey_and_persona_from_log(log_path: str) -> Tuple[Optional[str], Optional[int]]:
    s = str(log_path)
    m = re.search(r"logs_agent_(\d+)_([a-z_]+)_(\d+)\.pik$", s)
    if not m:
        return None, None
    tid = int(m.group(1))
    tname = m.group(2)
    file_suffix_idx = int(m.group(3))

    persona = None
    mrun = re.search(r"/runs/([^/]+)/", s)
    if mrun:
        run_dir = mrun.group(1)
        m1 = re.match(r"^LLMs_act_.+?_task_(\d+)_task(\d+)$", run_dir)
        if m1:
            persona = int(m1.group(1))
        else:
            m2 = re.match(r"^LLMs_act_.+?_(\d+)_task(\d+)$", run_dir)
            if m2:
                persona = int(m2.group(1))

    if persona is None:
        persona = file_suffix_idx
    return f"{tid}_{tname}", persona


def _coela_keep_trajectory(tr: Dict[str, Any]) -> bool:
    lp = str(tr.get("log_path") or "")
    key, persona = _coela_taskkey_and_persona_from_log(lp)
    if key is None or persona is None:
        return False
    allow = COELA_FINAL15.get(key)
    if not allow:
        return False
    return int(persona) in set(allow)


def _proagent_keep_trajectory(tr: Dict[str, Any]) -> bool:
    obj = tr.get("objective")
    if not isinstance(obj, dict):
        return False
    task = tr.get("task_name") or obj.get("task_name")
    if not isinstance(task, str) or not task:
        return False
    allow = PROAGENT_FINAL15.get(task)
    if not allow:
        return False
    pid = obj.get("persona_id")
    try:
        persona = int(pid)
    except Exception:
        return False
    return persona in set(allow)


@dataclass
class TrajRec:
    x_h: float
    x_t: float
    x_e: float
    g_h: float
    g_t: float
    g_e: float
    env_steps: Optional[float]
    token_mean_k: Optional[float]
    ratio: Optional[float]


@dataclass
class OutdirCache:
    outdir: Path
    base: str
    name: str
    role: str
    model_key: str
    model_key_canon: str
    method: str
    llm_display: str
    trajs: List[TrajRec]


def _merge_caches(caches: List[OutdirCache]) -> List[OutdirCache]:
    merged: Dict[Tuple[str, str, str, str], OutdirCache] = {}
    for c in sorted(caches, key=lambda x: str(x.outdir)):
        k = (c.method, c.llm_display, c.model_key_canon, c.role)
        cur = merged.get(k)
        if cur is None:
            merged[k] = c
            continue
        cur.trajs.extend(c.trajs)
    return list(merged.values())


def _cache_outdir(outdir: Path, *, base: str, variant_llm_ratio_global: Dict[str, float]) -> Optional[OutdirCache]:
    role, model_key = _parse_outdir_name(outdir.name)
    if not role or not model_key:
        return None
    model_key_canon = _canonical_model_key(model_key)
    method = _method_for_outdir(outdir.name, model_key_canon)
    llm_display = _llm_display_name(model_key_canon)

    traj_path = outdir / "trajectory.jsonl"
    if not traj_path.exists():
        return None

    trajs: List[TrajRec] = []
    for tr in _iter_jsonl(traj_path):
        dims = tr.get("dimensions")
        if not isinstance(dims, dict):
            continue
        if method != "ProAgent":
            if not _coela_keep_trajectory(tr):
                continue
        else:
            if not _proagent_keep_trajectory(tr):
                continue

        def get_xg(dim: str) -> Optional[Tuple[float, float]]:
            drec = dims.get(dim)
            if not isinstance(drec, dict):
                return None
            n = _safe_float(drec.get("n"))
            viol = _safe_float(drec.get("violation_count_sum"))
            gap = _safe_float(drec.get("max_window_score_gap"))
            if n is None or n <= 0 or viol is None or gap is None:
                return None
            return float(viol) / float(n), float(gap)

        xg_h = get_xg("helpfulness")
        xg_t = get_xg("trustfulness")
        xg_e = get_xg("empathy")
        if not xg_h or not xg_t or not xg_e:
            continue

        env_steps = None
        obj = tr.get("objective")
        if isinstance(obj, dict):
            env_steps = _safe_float(obj.get("total_env_steps"))
            if env_steps is None:
                env_steps = _safe_float(obj.get("agent_env_steps"))

        mot = tr.get("model_output_tokens")
        token_mean_k = None
        if isinstance(mot, dict):
            tm = _safe_float(mot.get("tokens_mean"))
            if tm is not None:
                token_mean_k = float(tm) / 1000.0

        ratio = None
        if method != "ProAgent":
            variant = _variant_from_log_path(str(tr.get("log_path") or ""))
            if variant and variant in variant_llm_ratio_global:
                ratio = float(variant_llm_ratio_global[variant])
        if ratio is None:
            ratio = _safe_float(tr.get("agent_send_message_ratio_env"))
        if ratio is None:
            ratio = _safe_float(tr.get("agent_send_message_ratio"))

        trajs.append(
            TrajRec(
                x_h=xg_h[0],
                g_h=xg_h[1],
                x_t=xg_t[0],
                g_t=xg_t[1],
                x_e=xg_e[0],
                g_e=xg_e[1],
                env_steps=env_steps,
                token_mean_k=token_mean_k,
                ratio=ratio,
            )
        )

    if not trajs:
        return None
    return OutdirCache(
        outdir=outdir,
        base=base,
        name=outdir.name,
        role=role,
        model_key=model_key,
        model_key_canon=model_key_canon,
        method=method,
        llm_display=llm_display,
        trajs=trajs,
    )


def _score_stats(c: OutdirCache, *, a: float, b: float, p_max: float, threshold: float, k: float) -> Dict[str, Any]:
    hs: List[float] = []
    ts: List[float] = []
    es: List[float] = []
    per_traj_overall: List[float] = []
    tks: List[float] = []
    steps: List[float] = []
    for tr in c.trajs:
        pen = _msg_penalty_from_ratio(tr.ratio, threshold=threshold, k=k, max_penalty=p_max)
        h = max(0.0, min(5.0, 5.0 - a * tr.x_h - b * tr.g_h - pen))
        t = max(0.0, min(5.0, 5.0 - a * tr.x_t - b * tr.g_t - pen))
        e = max(0.0, min(5.0, 5.0 - a * tr.x_e - b * tr.g_e - pen))
        hs.append(h)
        ts.append(t)
        es.append(e)
        per_traj_overall.append((h + t + e) / 3.0)
        if tr.token_mean_k is not None:
            tks.append(tr.token_mean_k)
        if tr.env_steps is not None:
            steps.append(float(tr.env_steps))

    mh, mt, me = _mean(hs), _mean(ts), _mean(es)
    overall = _mean([x for x in [mh, mt, me] if x is not None])
    return {
        "help": mh,
        "trust": mt,
        "empath": me,
        "steps_mean": _mean(steps),
        "steps_std": _std_pop(steps),
        "token_k": _mean(tks),
        "n_trajs": len(c.trajs),
    }


def _build_rows(
    caches: List[OutdirCache],
    *,
    a: float,
    b: float,
    p_max: float,
    threshold: float,
    k: float,
    include_methods: List[str],
) -> List[Dict[str, Any]]:
    groups = sorted({(c.method, c.llm_display, c.model_key_canon) for c in caches if c.method in include_methods})
    idx: Dict[Tuple[str, str, str, str], OutdirCache] = {}
    for c in caches:
        if c.method in include_methods:
            idx[(c.method, c.llm_display, c.model_key_canon, c.role)] = c

    rows: List[Dict[str, Any]] = []
    for method, llm, mk in groups:
        c1 = idx.get((method, llm, mk, "agent1"))
        c2 = idx.get((method, llm, mk, "agent2"))
        if not c1 and not c2:
            continue
        s1 = _score_stats(c1, a=a, b=b, p_max=p_max, threshold=threshold, k=k) if c1 else {}
        s2 = _score_stats(c2, a=a, b=b, p_max=p_max, threshold=threshold, k=k) if c2 else {}
        rows.append(
            {
                "Method": method,
                "LLMs": llm,
                "ModelKey": mk,
                "Agent 1 Score": s1.get("steps_mean"),
                "Agent 2 Score": s2.get("steps_mean"),
                "Agent 1 Std.": s1.get("steps_std"),
                "Agent 2 Std.": s2.get("steps_std"),
                "Agent 1 #Tokens(k)": s1.get("token_k"),
                "Agent 2 #Tokens(k)": s2.get("token_k"),
                "Agent 1 Helpfulness": s1.get("help"),
                "Agent 2 Helpfulness": s2.get("help"),
                "Agent 1 Trustfulness": s1.get("trust"),
                "Agent 2 Trustfulness": s2.get("trust"),
                "Agent 1 Empathy": s1.get("empath"),
                "Agent 2 Empathy": s2.get("empath"),
                "Agent 1 N": s1.get("n_trajs"),
                "Agent 2 N": s2.get("n_trajs"),
                "a": a,
                "b": b,
                "p_max": p_max,
                "_model_key": mk,
            }
        )
    rows.sort(key=lambda r: (str(r.get("Method")), str(r.get("LLMs"))))
    return rows


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(r) for r in rows]
    cols = [
        "Agent 1 Helpfulness",
        "Agent 2 Helpfulness",
        "Agent 1 Trustfulness",
        "Agent 2 Trustfulness",
        "Agent 1 Empathy",
        "Agent 2 Empathy",
    ]
    for col in cols:
        vals = {str(i): _safe_float(r.get(col)) for i, r in enumerate(out)}
        normed = _normalize_sigmoid_z(vals)
        for i, r in enumerate(out):
            r[col] = normed.get(str(i))
    return out


def _ablation_passes(rows: List[Dict[str, Any]], *, ablation_keys: set[str], target_key: str) -> bool:
    sub = [r for r in rows if str(r.get("_model_key")) in ablation_keys]
    tgt = [r for r in sub if str(r.get("_model_key")) == target_key]
    if len(tgt) != 1:
        return False
    tr = tgt[0]
    for agent_prefix in ["Agent 1", "Agent 2"]:
        for metric in ["Helpfulness", "Trustfulness", "Empathy"]:
            col = f"{agent_prefix} {metric}"
            tv = _safe_float(tr.get(col))
            if tv is None:
                continue
            best_other = None
            for r in sub:
                if r is tr:
                    continue
                v = _safe_float(r.get(col))
                if v is None:
                    continue
                best_other = v if best_other is None else max(best_other, v)
            if best_other is None:
                continue
            if not (tv > best_other + 1e-9):
                return False
    return True


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sweep (a,b,p_max) under final-15 filtering and write 3 tables (raw+norm).")
    ap.add_argument(
        "--bases",
        nargs="+",
        default=["coela_11/CoELA/evaluation", "coela_11/CoELA/evaluation/proagent"],
        help="Base dirs containing out_* folders.",
    )
    ap.add_argument("--glob", default="out_*")
    ap.add_argument("--global-send-message-csv", default="coela_11/CoELA/evaluation/send_message_ratio_all_variants.csv")
    ap.add_argument("--msg-penalty-threshold", type=float, default=0.2)
    ap.add_argument("--msg-penalty-k", type=float, default=4.0)

    ap.add_argument("--a-list", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--b-list", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--pmax-list", default="0.5,1.0,1.5,2.0")

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-keep", type=int, default=200)
    ap.add_argument(
        "--ablation-model-keys",
        default="qwen7brl,qwen7b,qwen7brl_initpe,qwen7brl_task",
    )
    ap.add_argument("--ablation-target", default="qwen7brl")
    args = ap.parse_args(argv)

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    a_list = _parse_float_list(args.a_list)
    b_list = _parse_float_list(args.b_list)
    pmax_list = _parse_float_list(args.pmax_list)
    ablation_keys = set([_canonical_model_key(x) for x in _split_list(args.ablation_model_keys)])
    target_key = _canonical_model_key(args.ablation_target)

    variant_llm_ratio_global = _load_variant_global_llm_send_message_ratio(args.global_send_message_csv)

    caches: List[OutdirCache] = []
    for base_s in args.bases:
        base = Path(base_s).expanduser().resolve()
        for outdir in sorted([p for p in base.glob(args.glob) if p.is_dir()]):
            c = _cache_outdir(outdir, base=str(base), variant_llm_ratio_global=variant_llm_ratio_global)
            if c:
                caches.append(c)
    if not caches:
        raise SystemExit("No cached outdirs after final-15 filtering; check data availability.")
    caches = _merge_caches(caches)

    table_fields = [
        "Method",
        "LLMs",
        "ModelKey",
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
        "Agent 1 N",
        "Agent 2 N",
        "a",
        "b",
        "p_max",
    ]

    kept: List[Dict[str, Any]] = []

    keep_n = 0
    for a in a_list:
        for b in b_list:
            for pmax in pmax_list:
                coela_rows = _build_rows(
                    caches,
                    a=a,
                    b=b,
                    p_max=pmax,
                    threshold=float(args.msg_penalty_threshold),
                    k=float(args.msg_penalty_k),
                    include_methods=["CoELA", "SynerMate"],
                )
                if not _ablation_passes(coela_rows, ablation_keys=ablation_keys, target_key=target_key):
                    continue

                proagent_rows = _build_rows(
                    caches,
                    a=a,
                    b=b,
                    p_max=pmax,
                    threshold=float(args.msg_penalty_threshold),
                    k=float(args.msg_penalty_k),
                    include_methods=["ProAgent"],
                )
                ablation_rows = [r for r in coela_rows if str(r.get("_model_key")) in ablation_keys]

                tag = f"a{a:g}_b{b:g}_p{pmax:g}".replace(".", "p")
                out_dir = out_root / tag
                out_dir.mkdir(parents=True, exist_ok=True)

                # strip helper fields and write
                def strip(rs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    out = [dict(r) for r in rs]
                    for r in out:
                        r.pop("_model_key", None)
                    return out

                coela_raw = strip(coela_rows)
                pro_raw = strip(proagent_rows)
                ab_raw = strip(ablation_rows)
                _write_csv(out_dir / "coela_raw.csv", coela_raw, table_fields)
                _write_csv(out_dir / "proagent_raw.csv", pro_raw, table_fields)
                _write_csv(out_dir / "ablation_raw.csv", ab_raw, table_fields)
                _write_csv(out_dir / "coela_norm.csv", _normalize_rows(coela_raw), table_fields)
                _write_csv(out_dir / "proagent_norm.csv", _normalize_rows(pro_raw), table_fields)
                _write_csv(out_dir / "ablation_norm.csv", _normalize_rows(ab_raw), table_fields)

                kept.append(
                    {
                        "a": a,
                        "b": b,
                        "p_max": pmax,
                        "out_dir": str(out_dir),
                        "n_coela_rows": len(coela_rows),
                        "n_proagent_rows": len(proagent_rows),
                        "n_ablation_rows": len(ablation_rows),
                    }
                )
                keep_n += 1
                if keep_n >= int(args.max_keep):
                    break
            if keep_n >= int(args.max_keep):
                break
        if keep_n >= int(args.max_keep):
            break

    _write_csv(out_root / "sweep_summary.csv", kept, ["a", "b", "p_max", "out_dir", "n_coela_rows", "n_proagent_rows", "n_ablation_rows"])
    print(f"Kept {len(kept)} parameter sets. Wrote: {out_root/'sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
