#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN"
    message: str
    file: Optional[str] = None


@dataclass
class FileReport:
    path: str
    issues: List[Issue]
    stats: Dict[str, Any]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _find_latest_run_dir(runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _list_log_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("logs_agent_*.pik"))


def _contains_unreplaced_placeholders(text: str) -> bool:
    # 典型占位符：$GOAL / $PROGRESS / $AGENT_NAME / {AGENT_NAME} 等
    return bool(re.search(r"(\$[A-Z_]+|\{[A-Z_]+\})", text))


def _check_prompt_basic(raw_prompt: str, agent_name: str, oppo_name: str) -> List[Issue]:
    issues: List[Issue] = []
    if agent_name and agent_name not in raw_prompt:
        issues.append(Issue("WARN", f"raw_prompt 中未出现 agent_name={agent_name!r}"))
    if oppo_name and oppo_name not in raw_prompt:
        issues.append(Issue("WARN", f"raw_prompt 中未出现 oppo_name={oppo_name!r}"))
    if _contains_unreplaced_placeholders(raw_prompt):
        issues.append(Issue("WARN", "raw_prompt 仍包含未替换的占位符（如 $GOAL/{AGENT_NAME}）"))
    return issues


def _summarize_lengths(saved_info: Dict[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for key in ("action", "plan", "LLM", "obs", "belief", "belief_graph", "graph"):
        if key not in saved_info:
            continue
        val = saved_info[key]
        if isinstance(val, dict):
            stats[f"len_{key}_0"] = len(val.get(0, []))
            stats[f"len_{key}_1"] = len(val.get(1, []))
        elif isinstance(val, list):
            stats[f"len_{key}"] = len(val)
    stats["finished"] = saved_info.get("finished", None)
    return stats


def audit_log_file(path: Path, max_steps_check: int = 5) -> FileReport:
    issues: List[Issue] = []
    stats: Dict[str, Any] = {}

    try:
        saved_info = _load_pickle(path)
    except Exception as e:
        issues.append(Issue("ERROR", f"无法读取 pik：{e}", file=str(path)))
        return FileReport(path=str(path), issues=issues, stats=stats)

    if not isinstance(saved_info, dict):
        issues.append(Issue("ERROR", f"pik 内容不是 dict（实际：{type(saved_info)}）", file=str(path)))
        return FileReport(path=str(path), issues=issues, stats=stats)

    for required_key in ("task_id", "task_name", "action", "plan", "LLM", "finished"):
        if required_key not in saved_info:
            issues.append(Issue("ERROR", f"缺少字段 {required_key!r}", file=str(path)))

    stats.update(_summarize_lengths(saved_info))

    # 基本一致性：两个 agent 的 action/plan/LLM 长度不应为 0
    for key in ("action", "plan", "LLM"):
        val = saved_info.get(key)
        if isinstance(val, dict):
            if len(val.get(0, [])) == 0:
                issues.append(Issue("WARN", f"{key}[0] 为空", file=str(path)))
            if len(val.get(1, [])) == 0:
                issues.append(Issue("WARN", f"{key}[1] 为空", file=str(path)))

    # 检查 LLM 记录是否包含 raw_prompt/raw_output（我们已在 LLM.run() 里加入）
    llm = saved_info.get("LLM")
    if isinstance(llm, dict):
        for agent_idx in (0, 1):
            entries = llm.get(agent_idx, [])
            if not entries:
                continue
            first_entries = entries[:max_steps_check]
            found_prompt = False
            for entry in first_entries:
                if not isinstance(entry, dict):
                    continue
                raw_prompt = entry.get("raw_prompt")
                raw_output = entry.get("raw_output")
                if raw_prompt is not None and isinstance(raw_prompt, str) and raw_prompt.strip():
                    found_prompt = True
                    agent_name = str(entry.get("agent_name", "") or "")
                    oppo_name = str(entry.get("oppo_name", "") or "")
                    issues.extend(_check_prompt_basic(raw_prompt, agent_name, oppo_name))
                if raw_output is None or not isinstance(raw_output, str) or not raw_output.strip():
                    issues.append(Issue("WARN", f"LLM[{agent_idx}] 缺少 raw_output 或为空（step 内）", file=str(path)))
                else:
                    # 输出里最好包含 <action>，否则解析可能不稳
                    if "<action" not in raw_output.lower():
                        issues.append(Issue("WARN", f"LLM[{agent_idx}] raw_output 未包含 <action> 标签（可能影响解析）", file=str(path)))
            if not found_prompt:
                issues.append(Issue("WARN", f"LLM[{agent_idx}] 前 {max_steps_check} 步未发现 raw_prompt", file=str(path)))

    return FileReport(path=str(path), issues=issues, stats=stats)


def dump_raw_text(saved_info: Dict[str, Any], out_path: Path, max_steps: int = 0) -> None:
    """
    将每一步 LLM 的 raw_prompt/raw_output 等原始文本导出为可人工查看的 txt。
    max_steps=0 表示导出全部步。
    """
    llm = saved_info.get("LLM")
    if not isinstance(llm, dict):
        out_path.write_text("[dump] saved_info['LLM'] 不存在或格式不正确\n", encoding="utf-8")
        return

    task_id = saved_info.get("task_id")
    task_name = saved_info.get("task_name")
    env_id = saved_info.get("env_id")

    lines: List[str] = []
    lines.append(f"[meta] task_id={task_id} task_name={task_name} env_id={env_id}\n")

    for agent_idx in (0, 1):
        entries = llm.get(agent_idx, [])
        if not isinstance(entries, list):
            continue
        lines.append(f"\n========== agent_index={agent_idx} steps={len(entries)} ==========\n")
        take = entries if max_steps == 0 else entries[:max_steps]
        for step, entry in enumerate(take):
            if not isinstance(entry, dict):
                lines.append(f"\n--- step={step} (invalid entry type={type(entry)}) ---\n")
                continue
            agent_name = entry.get("agent_name")
            oppo_name = entry.get("oppo_name")
            plan = entry.get("plan")
            think = entry.get("think")
            message = entry.get("message")
            raw_prompt = entry.get("raw_prompt")
            raw_output = entry.get("raw_output")

            lines.append(f"\n--- step={step} agent_name={agent_name} oppo_name={oppo_name} ---\n")
            lines.append(f"[plan]\n{plan}\n")
            lines.append(f"[think]\n{think}\n")
            lines.append(f"[message]\n{message}\n")
            lines.append("\n[raw_prompt]\n")
            lines.append((raw_prompt or "") + "\n")
            lines.append("\n[raw_output]\n")
            lines.append((raw_output or "") + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def dump_raw_jsonl(saved_info: Dict[str, Any], out_path: Path, max_steps: int = 0) -> None:
    """
    将每一步 LLM 的原始字段导出为 JSONL（便于后处理/grep）。
    max_steps=0 表示导出全部步。
    """
    llm = saved_info.get("LLM")
    if not isinstance(llm, dict):
        out_path.write_text("", encoding="utf-8")
        return

    task_id = saved_info.get("task_id")
    task_name = saved_info.get("task_name")
    env_id = saved_info.get("env_id")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for agent_idx in (0, 1):
            entries = llm.get(agent_idx, [])
            if not isinstance(entries, list):
                continue
            take = entries if max_steps == 0 else entries[:max_steps]
            for step, entry in enumerate(take):
                if not isinstance(entry, dict):
                    continue
                payload = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "env_id": env_id,
                    "agent_index": agent_idx,
                    "step": step,
                    "agent_name": entry.get("agent_name"),
                    "oppo_name": entry.get("oppo_name"),
                    "plan": entry.get("plan"),
                    "think": entry.get("think"),
                    "message": entry.get("message"),
                    "raw_prompt": entry.get("raw_prompt"),
                    "raw_output": entry.get("raw_output"),
                    "raw_usage": entry.get("raw_usage"),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def audit_run_dir(run_dir: Path, max_files: int = 0, max_steps_check: int = 5) -> Tuple[List[Issue], List[FileReport]]:
    issues: List[Issue] = []
    reports: List[FileReport] = []

    if not run_dir.exists():
        issues.append(Issue("ERROR", "run_dir 不存在", file=str(run_dir)))
        return issues, reports

    console = run_dir / "console.txt"
    if not console.exists():
        issues.append(Issue("WARN", "未找到 console.txt（如果你是旧脚本运行，这是正常的）", file=str(console)))

    results = run_dir / "results.pik"
    if not results.exists():
        issues.append(Issue("WARN", "未找到 results.pik", file=str(results)))

    logpik = run_dir / "log.pik"
    if not logpik.exists():
        issues.append(Issue("WARN", "未找到 log.pik", file=str(logpik)))

    log_files = _list_log_files(run_dir)
    if not log_files:
        issues.append(Issue("ERROR", "未找到 logs_agent_*.pik", file=str(run_dir)))
        return issues, reports

    if max_files and len(log_files) > max_files:
        log_files = log_files[:max_files]
        issues.append(Issue("WARN", f"只审查前 {max_files} 个 logs_agent_*.pik（其余已跳过）", file=str(run_dir)))

    for lf in log_files:
        reports.append(audit_log_file(lf, max_steps_check=max_steps_check))

    return issues, reports


def _print_summary(run_dir: Path, issues: List[Issue], reports: List[FileReport]) -> int:
    err = sum(1 for i in issues if i.level == "ERROR")
    warn = sum(1 for i in issues if i.level == "WARN")
    for r in reports:
        err += sum(1 for i in r.issues if i.level == "ERROR")
        warn += sum(1 for i in r.issues if i.level == "WARN")

    print(f"[audit] run_dir={run_dir}")
    print(f"[audit] files={len(reports)} errors={err} warnings={warn}")

    def _emit(issue: Issue):
        where = f" file={issue.file}" if issue.file else ""
        print(f"[{issue.level}] {issue.message}{where}")

    for i in issues:
        _emit(i)
    for r in reports:
        for i in r.issues:
            _emit(i)

    # 每个文件打印一个简短 stats 行，方便你快速对照“日志是否完整”
    for r in reports:
        s = r.stats
        brief = {
            "finished": s.get("finished"),
            "action0": s.get("len_action_0"),
            "action1": s.get("len_action_1"),
            "llm0": s.get("len_LLM_0"),
            "llm1": s.get("len_LLM_1"),
            "obs0": s.get("len_obs_0"),
            "obs1": s.get("len_obs_1"),
        }
        print(f"[stats] {r.path} {brief}")

    return 0 if err == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CWAH run outputs for prompt/log completeness.")
    parser.add_argument("--runs-root", type=str, default=None, help="runs 根目录（默认：<repo>/runs）")
    parser.add_argument("--run-dir", type=str, default=None, help="指定某次运行目录（如 runs/LLMs_act_xxx）")
    parser.add_argument("--max-files", type=int, default=0, help="最多审查多少个 logs_agent_*.pik（0=全部）")
    parser.add_argument("--max-steps-check", type=int, default=5, help="每个 agent 最多检查前 N 步 LLM 记录")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="导出原始文本/JSONL 的目录（默认：<run_dir>/audit_dump）",
    )
    parser.add_argument("--dump-max-steps", type=int, default=0, help="导出每个 agent 前 N 步（0=全部）")
    parser.add_argument("--dump-format", choices=["txt", "jsonl", "both", "none"], default="both", help="导出格式（none=不导出）")
    parser.add_argument("--json-out", type=str, default=None, help="将审查结果写入 JSON 文件")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = Path(args.runs_root) if args.runs_root else (repo_root / "runs")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
    else:
        run_dir = _find_latest_run_dir(runs_root)
        if run_dir is None:
            print(f"[ERROR] runs_root 下未找到任何运行目录：{runs_root}")
            return 2

    issues, reports = audit_run_dir(run_dir, max_files=args.max_files, max_steps_check=args.max_steps_check)
    exit_code = _print_summary(run_dir, issues, reports)

    dump_dir_arg = args.dump_dir
    if dump_dir_arg is None:
        dump_root = run_dir / "audit_dump"
    else:
        dump_root = Path(dump_dir_arg)
        if not dump_root.is_absolute():
            dump_root = run_dir / dump_root

    if args.dump_format != "none":
        dump_root.mkdir(parents=True, exist_ok=True)
        for report in reports:
            log_path = Path(report.path)
            try:
                saved_info = _load_pickle(log_path)
            except Exception:
                continue
            stem = log_path.stem
            if args.dump_format in ("txt", "both"):
                dump_raw_text(saved_info, dump_root / f"{stem}.txt", max_steps=args.dump_max_steps)
            if args.dump_format in ("jsonl", "both"):
                dump_raw_jsonl(saved_info, dump_root / f"{stem}.jsonl", max_steps=args.dump_max_steps)
        print(f"[audit] dumped_raw_to={dump_root}")

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        payload = {
            "run_dir": str(run_dir),
            "issues": [asdict(i) for i in issues],
            "reports": [
                {"path": r.path, "issues": [asdict(i) for i in r.issues], "stats": r.stats} for r in reports
            ],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[audit] wrote_json={out_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
