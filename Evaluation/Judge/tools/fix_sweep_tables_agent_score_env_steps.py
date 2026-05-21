#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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


@dataclass(frozen=True)
class Key:
    method: str
    llm: str
    role: str  # agent1/agent2


def _load_helpers() -> Any:
    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir))
    import llmjudge_sweep_tables_final15_fast as sweep  # type: ignore

    return sweep


def _env_steps_from_tr(tr: Dict[str, Any]) -> Optional[float]:
    obj = tr.get("objective")
    if not isinstance(obj, dict):
        return None
    v = _safe_float(obj.get("total_env_steps"))
    if v is None:
        v = _safe_float(obj.get("agent_env_steps"))
    return v


def _build_steps_stats_by_key(
    *,
    bases: List[Path],
    glob: str,
) -> Dict[Key, Tuple[Optional[float], Optional[float]]]:
    sweep = _load_helpers()

    stats: Dict[Key, List[float]] = {}
    for base in bases:
        outdirs = sorted([p for p in base.glob(glob) if p.is_dir()])
        for outdir in outdirs:
            role, model_key = sweep._parse_outdir_name(outdir.name)  # noqa: SLF001
            if not role or not model_key:
                continue
            model_key_canon = sweep._canonical_model_key(model_key)  # noqa: SLF001
            method = sweep._method_for_outdir(outdir.name, model_key_canon)  # noqa: SLF001
            llm = sweep._llm_display_name(model_key_canon)  # noqa: SLF001

            traj_path = outdir / "trajectory.jsonl"
            if not traj_path.exists():
                continue

            steps: List[float] = []
            for tr in _iter_jsonl(traj_path):
                if method != "ProAgent":
                    if not sweep._coela_keep_trajectory(tr):  # noqa: SLF001
                        continue
                else:
                    if not sweep._proagent_keep_trajectory(tr):  # noqa: SLF001
                        continue

                es = _env_steps_from_tr(tr)
                if es is not None:
                    steps.append(float(es))

            if not steps:
                continue
            k = Key(method=method, llm=llm, role=role)
            stats.setdefault(k, []).extend(steps)

    out: Dict[Key, Tuple[Optional[float], Optional[float]]] = {}
    for k, xs in stats.items():
        out[k] = (_mean(xs), _std_pop(xs))
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{float(v):.6f}".rstrip("0").rstrip(".")


def _update_one_table(path: Path, stats: Dict[Key, Tuple[Optional[float], Optional[float]]]) -> bool:
    rows = _read_csv(path)
    if not rows:
        return False
    fieldnames = list(rows[0].keys())
    if "Method" not in fieldnames or "LLMs" not in fieldnames:
        return False
    if "Agent 1 Score" not in fieldnames or "Agent 2 Score" not in fieldnames:
        return False

    changed = False
    for row in rows:
        method = (row.get("Method") or "").strip()
        llm = (row.get("LLMs") or "").strip()
        for agent_idx, role in [(1, "agent1"), (2, "agent2")]:
            k = Key(method=method, llm=llm, role=role)
            mean_std = stats.get(k)
            if not mean_std:
                continue
            mean_v, std_v = mean_std
            score_key = f"Agent {agent_idx} Score"
            std_key = f"Agent {agent_idx} Std."
            new_score = _fmt(mean_v)
            new_std = _fmt(std_v)
            if row.get(score_key, "") != new_score:
                row[score_key] = new_score
                changed = True
            if std_key in row and row.get(std_key, "") != new_std:
                row[std_key] = new_std
                changed = True

    if changed:
        _write_csv(path, rows, fieldnames)
    return changed


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rewrite existing sweep tables so `Agent X Score/Std.` reflect mean/std total_env_steps (final-15 filtered)."
    )
    ap.add_argument(
        "--sweep-dir",
        required=True,
        help="Sweep directory containing per-parameter subdirs (e.g., .../final15_sweep_qwen7_ab_step0p1).",
    )
    ap.add_argument(
        "--bases",
        nargs="+",
        default=["Evaluation/evaluation", "Evaluation/Judge/cook"],
        help="Base dirs containing out_* folders (used to compute env-step stats).",
    )
    ap.add_argument("--glob", default="out_*")
    args = ap.parse_args(argv)

    sweep_dir = Path(os.path.expanduser(args.sweep_dir)).resolve()
    bases = [Path(os.path.expanduser(b)).resolve() for b in args.bases]

    stats = _build_steps_stats_by_key(bases=bases, glob=str(args.glob))

    n_tables = 0
    n_changed = 0
    for p in sorted(sweep_dir.iterdir()):
        if not (p.is_dir() and p.name.startswith("a")):
            continue
        for name in [
            "coela_raw.csv",
            "coela_norm.csv",
            "proagent_raw.csv",
            "proagent_norm.csv",
            "ablation_raw.csv",
            "ablation_norm.csv",
        ]:
            table = p / name
            if not table.exists():
                continue
            n_tables += 1
            if _update_one_table(table, stats):
                n_changed += 1

    print(f"Updated {n_changed}/{n_tables} tables under {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
