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
    mk = re.sub(r"([_-])part\d+$", "", model_key, flags=re.IGNORECASE)
    mkl = mk.lower()
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
    if m in ("qwen7b",):
        return "Qwen2.5-7B-Instruct"
    if m in ("qwen7brl",):
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
    out: List[float] = []
    for x in _split_list(s):
        out.append(float(x))
    return out


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
    token_k: Optional[float]
    ratio_global: Optional[float]
    # Per-trajectory arrays (same length)
    x_h: List[float]
    x_t: List[float]
    x_e: List[float]
    g_h: List[float]
    g_t: List[float]
    g_e: List[float]
    env_steps: List[float]


def _load_token_k_by_outdir_from_tuning_means(tuning_means_csv: Path) -> Dict[str, float]:
    if not tuning_means_csv.exists():
        return {}
    out: Dict[str, float] = {}
    with tuning_means_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            outdir = (row.get("outdir") or "").strip()
            tm = _safe_float(row.get("model_output_tokens_mean_mean"))
            if outdir and tm is not None:
                out[outdir] = float(tm) / 1000.0
    return out


def _cache_outdir(
    outdir: Path,
    *,
    base: str,
    variant_llm_ratio_global: Dict[str, float],
    token_k_by_outdir: Dict[str, float],
) -> Optional[OutdirCache]:
    role, model_key = _parse_outdir_name(outdir.name)
    if not role or not model_key:
        return None
    model_key_canon = _canonical_model_key(model_key)
    method = _method_for_outdir(outdir.name, model_key_canon)
    llm_display = _llm_display_name(model_key_canon)

    traj_path = outdir / "trajectory.jsonl"
    if not traj_path.exists():
        return None

    # Global ratio per variant (CWAH), else fallback to per-trajectory field (but we still store None here).
    ratio_global = None
    for tr in _iter_jsonl(traj_path):
        log_path = str(tr.get("log_path") or "")
        variant = _variant_from_log_path(log_path)
        if variant and variant in variant_llm_ratio_global:
            ratio_global = float(variant_llm_ratio_global[variant])
        break

    x_h: List[float] = []
    x_t: List[float] = []
    x_e: List[float] = []
    g_h: List[float] = []
    g_t: List[float] = []
    g_e: List[float] = []
    env_steps: List[float] = []

    for tr in _iter_jsonl(traj_path):
        dims = tr.get("dimensions")
        if not isinstance(dims, dict):
            continue
        ok = True
        vals: Dict[str, Tuple[float, float]] = {}
        for dim, xacc, gacc in [
            ("helpfulness", x_h, g_h),
            ("trustfulness", x_t, g_t),
            ("empathy", x_e, g_e),
        ]:
            drec = dims.get(dim)
            if not isinstance(drec, dict):
                ok = False
                break
            n = _safe_float(drec.get("n"))
            viol = _safe_float(drec.get("violation_count_sum"))
            gap = _safe_float(drec.get("max_window_score_gap"))
            if n is None or n <= 0 or viol is None or gap is None:
                ok = False
                break
            vals[dim] = (float(viol) / float(n), float(gap))
        if not ok:
            continue
        x_h.append(vals["helpfulness"][0])
        g_h.append(vals["helpfulness"][1])
        x_t.append(vals["trustfulness"][0])
        g_t.append(vals["trustfulness"][1])
        x_e.append(vals["empathy"][0])
        g_e.append(vals["empathy"][1])
        obj = tr.get("objective")
        if isinstance(obj, dict):
            es = _safe_float(obj.get("total_env_steps"))
            if es is None:
                es = _safe_float(obj.get("agent_env_steps"))
            if es is not None:
                env_steps.append(float(es))

    if not x_h:
        return None

    token_k = token_k_by_outdir.get(str(outdir))
    return OutdirCache(
        outdir=outdir,
        base=base,
        name=outdir.name,
        role=role,
        model_key=model_key,
        model_key_canon=model_key_canon,
        method=method,
        llm_display=llm_display,
        token_k=token_k,
        ratio_global=ratio_global,
        x_h=x_h,
        x_t=x_t,
        x_e=x_e,
        g_h=g_h,
        g_t=g_t,
        g_e=g_e,
        env_steps=env_steps,
    )


def _score_list(x: List[float], g: List[float], *, a: float, b: float, penalty: float) -> List[float]:
    # score per trajectory, clipped.
    out: List[float] = []
    for xi, gi in zip(x, g):
        s = 5.0 - a * float(xi) - b * float(gi) - float(penalty or 0.0)
        out.append(max(0.0, min(5.0, s)))
    return out


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


def _write_flat_table(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sweep (a,b,p_max) and export paper-style tables quickly from cached trajectory.jsonl."
    )
    ap.add_argument(
        "--bases",
        nargs="+",
        default=["Evaluation/evaluation", "Evaluation/Judge/cook"],
        help="One or more base dirs containing out_* folders.",
    )
    ap.add_argument("--glob", default="out_*", help='Outdir glob under each base (default: "out_*").')
    ap.add_argument("--tuning-means-csv", default="Evaluation/Judge/tools/tuning_means.csv")
    ap.add_argument("--global-send-message-csv", default="Evaluation/Judge/tools/send_message_ratio_all_variants.csv")
    ap.add_argument("--msg-penalty-threshold", type=float, default=0.15)
    ap.add_argument("--msg-penalty-k", type=float, default=4.0)

    ap.add_argument("--a-list", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--b-list", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--pmax-list", default="0.5,1.0,1.5,2.0")

    ap.add_argument("--out-dir", default="Evaluation/sweeps/fast_tables")
    ap.add_argument("--max-keep", type=int, default=200, help="Max parameter sets to keep (write tables for).")

    ap.add_argument(
        "--ablation-model-keys",
        default="qwen7brl,qwen7b,qwen7brl_initpe,qwen7brl_task",
        help="Comma list of canonical model keys to include in ablation table.",
    )
    ap.add_argument("--ablation-target", default="qwen7brl", help="Canonical model key that must be best on all 3 dims.")
    args = ap.parse_args(argv)

    a_list = _parse_float_list(args.a_list)
    b_list = _parse_float_list(args.b_list)
    pmax_list = _parse_float_list(args.pmax_list)
    ablation_keys = set([_canonical_model_key(x) for x in _split_list(args.ablation_model_keys)])
    target_key = _canonical_model_key(args.ablation_target)

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    variant_llm_ratio_global = _load_variant_global_llm_send_message_ratio(args.global_send_message_csv)
    token_k_by_outdir = _load_token_k_by_outdir_from_tuning_means(Path(args.tuning_means_csv).expanduser().resolve())

    caches: List[OutdirCache] = []
    for base_s in args.bases:
        base = Path(base_s).expanduser().resolve()
        for outdir in sorted([p for p in base.glob(args.glob) if p.is_dir()]):
            c = _cache_outdir(
                outdir,
                base=str(base),
                variant_llm_ratio_global=variant_llm_ratio_global,
                token_k_by_outdir=token_k_by_outdir,
            )
            if c:
                caches.append(c)

    if not caches:
        raise SystemExit("No outdirs cached. Check --bases/--glob.")

    # Group caches into paper rows by (method, llm_display, model_key_canon), with agent1/agent2 slots.
    group_keys = sorted({(c.method, c.llm_display, c.model_key_canon) for c in caches})
    # Index caches by (method,llm,canon,role)
    idx: Dict[Tuple[str, str, str, str], OutdirCache] = {}
    for c in caches:
        idx[(c.method, c.llm_display, c.model_key_canon, c.role)] = c

    kept: List[Dict[str, Any]] = []

    # Fixed schema for the exported tables.
    table_fields = [
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
        "a",
        "b",
        "p_max",
    ]

    def build_rows_for_param(a: float, b: float, p_max: float, *, method_filter: Optional[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for method, llm, mk in group_keys:
            if method_filter == "COELA_ONLY":
                if method == "ProAgent":
                    continue
            elif method_filter:
                if method != method_filter:
                    continue
            c1 = idx.get((method, llm, mk, "agent1"))
            c2 = idx.get((method, llm, mk, "agent2"))
            if not c1 and not c2:
                continue

            def agg(c: Optional[OutdirCache]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
                if not c:
                    return None, None, None, None, None
                ratio = c.ratio_global
                pen = _msg_penalty_from_ratio(
                    ratio,
                    threshold=float(args.msg_penalty_threshold),
                    k=float(args.msg_penalty_k),
                    max_penalty=float(p_max),
                )
                sh = _score_list(c.x_h, c.g_h, a=a, b=b, penalty=pen)
                st = _score_list(c.x_t, c.g_t, a=a, b=b, penalty=pen)
                se = _score_list(c.x_e, c.g_e, a=a, b=b, penalty=pen)
                mh = _mean(sh)
                mt = _mean(st)
                me = _mean(se)
                steps_mean = _mean(c.env_steps)
                steps_std = _std_pop(c.env_steps) if c.env_steps else None
                return steps_mean, steps_std, mh, mt, me

            s1, std1, h1, t1, e1 = agg(c1)
            s2, std2, h2, t2, e2 = agg(c2)
            rows.append(
                {
                    "Method": method,
                    "LLMs": llm,
                    "Agent 1 Score": s1,
                    "Agent 2 Score": s2,
                    "Agent 1 Std.": std1,
                    "Agent 2 Std.": std2,
                    "Agent 1 #Tokens(k)": (c1.token_k if c1 else None),
                    "Agent 2 #Tokens(k)": (c2.token_k if c2 else None),
                    "Agent 1 Helpfulness": h1,
                    "Agent 2 Helpfulness": h2,
                    "Agent 1 Trustfulness": t1,
                    "Agent 2 Trustfulness": t2,
                    "Agent 1 Empathy": e1,
                    "Agent 2 Empathy": e2,
                    "a": a,
                    "b": b,
                    "p_max": p_max,
                    "_model_key": mk,
                }
            )
        # Keep a stable ordering similar to the paper: method then llm
        rows.sort(key=lambda r: (str(r.get("Method")), str(r.get("LLMs"))))
        return rows

    def normalize_affective(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Normalize the 6 affective columns with sigmoid(zscore) within this table.
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

    def ablation_passes(rows: List[Dict[str, Any]]) -> bool:
        # Check within ablation subset, target must be top on all 3 dims for both agents if present.
        sub = [r for r in rows if str(r.get("_model_key")) in ablation_keys]
        if not sub:
            return False
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

    keep_n = 0
    for a in a_list:
        for b in b_list:
            for p_max in pmax_list:
                coela_rows = build_rows_for_param(a, b, p_max, method_filter="COELA_ONLY")
                proagent_rows = build_rows_for_param(a, b, p_max, method_filter="ProAgent")
                ablation_rows = [r for r in coela_rows if str(r.get("_model_key")) in ablation_keys]

                if not ablation_passes(coela_rows):
                    continue

                tag = f"a{a:g}_b{b:g}_p{p_max:g}".replace(".", "p")
                out_dir = out_root / tag
                out_dir.mkdir(parents=True, exist_ok=True)

                # CoELA-only tables (includes SynerMate)
                coela_raw = [dict(r) for r in coela_rows]
                for r in coela_raw:
                    r.pop("_model_key", None)
                _write_flat_table(out_dir / "coela_raw.csv", coela_raw, table_fields)
                coela_norm = normalize_affective(coela_raw)
                _write_flat_table(out_dir / "coela_norm.csv", coela_norm, table_fields)

                # ProAgent-only tables
                pa_raw = [dict(r) for r in proagent_rows]
                for r in pa_raw:
                    r.pop("_model_key", None)
                _write_flat_table(out_dir / "proagent_raw.csv", pa_raw, table_fields)
                pa_norm = normalize_affective(pa_raw)
                _write_flat_table(out_dir / "proagent_norm.csv", pa_norm, table_fields)

                # Ablation tables (CoELA-only subset)
                ab_raw = [dict(r) for r in ablation_rows]
                for r in ab_raw:
                    r.pop("_model_key", None)
                _write_flat_table(out_dir / "ablation_raw.csv", ab_raw, table_fields)
                ab_norm = normalize_affective(ab_raw)
                _write_flat_table(out_dir / "ablation_norm.csv", ab_norm, table_fields)

                kept.append(
                    {
                        "a": a,
                        "b": b,
                        "p_max": p_max,
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

    summary_path = out_root / "sweep_summary.csv"
    summary_fields = ["a", "b", "p_max", "out_dir", "n_coela_rows", "n_proagent_rows", "n_ablation_rows"]
    _write_flat_table(summary_path, kept, summary_fields)
    print(f"Kept {len(kept)} parameter sets. Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
