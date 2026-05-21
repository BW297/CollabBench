#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _dumps_compact(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _to_dialogue_text(conversation_history: Any) -> str:
    if isinstance(conversation_history, str):
        return conversation_history
    if not isinstance(conversation_history, list):
        return _dumps_compact(conversation_history)
    parts: List[str] = []
    for m in conversation_history:
        if isinstance(m, str):
            parts.append(m)
        elif isinstance(m, dict):
            role = str(m.get("role", "")).strip()
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if role:
                parts.append(f"{role}: {content}")
            else:
                parts.append(content)
        else:
            parts.append(str(m))
    return "\n".join(parts)


def _build_step_index(entries: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
    pairs: List[Tuple[int, Dict[str, Any]]] = []
    for e in entries:
        t = _safe_int(e.get("timestep"), -1)
        if t >= 0:
            pairs.append((t, e))
    pairs.sort(key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _latest_entry_at_or_before(step_index: Tuple[List[int], List[Dict[str, Any]]], timestep: int) -> Optional[Dict[str, Any]]:
    steps, vals = step_index
    if not steps:
        return None
    if timestep < 0:
        return vals[-1]
    lo, hi = 0, len(steps)
    while lo < hi:
        mid = (lo + hi) // 2
        if steps[mid] <= timestep:
            lo = mid + 1
        else:
            hi = mid
    pos = lo - 1
    if pos < 0:
        return vals[0]
    return vals[pos]


def _parse_agent_human_ids_from_root(data_root: Path) -> Tuple[Optional[int], Optional[int]]:
    name = data_root.name
    if name == "actions":
        name = data_root.parent.name
    m = re.match(r"^src-(\d+)-agent-(\d+)-human", name)
    if not m:
        m = re.match(r"^src-(\d+)-human-(\d+)-agent", name)
        if not m:
            return None, None
        human_id = int(m.group(1))
        agent_id = int(m.group(2))
        return agent_id, human_id
    agent_id = int(m.group(1))
    human_id = int(m.group(2))
    return agent_id, human_id


def _parse_task_and_persona(folder_name: str) -> Tuple[str, Optional[int], Optional[int]]:
    k: Optional[int] = None
    persona: Optional[int] = None
    task = ""

    m = re.match(r"^actions_(\d+)_400_(.+?)_qwen2\.5_l2", folder_name)
    if m:
        k = int(m.group(1))
        task = m.group(2)
    else:
        task = folder_name

    m2 = re.search(r"using_big_5_True_(\d+)$", folder_name)
    if m2:
        persona = int(m2.group(1))
    return task, k, persona


def _iter_actions_json(data_root: Path, *, max_files: Optional[int] = None) -> Iterable[Path]:
    actions_dir = data_root / "actions"
    if actions_dir.exists():
        base = actions_dir
    else:
        base = data_root
    paths = sorted(base.rglob("step_1/actions.json"))
    if max_files is not None:
        paths = paths[: max_files]
    for p in paths:
        yield p


def cmd_build_windows(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build CoELA-compatible windows.jsonl from ProAgent step_1/actions.json files."
    )
    ap.add_argument(
        "--data-root",
        required=True,
        help="ProAgent experiment root (e.g. .../src-0-agent-1-human-...); searches actions/**/step_1/actions.json",
    )
    ap.add_argument("--out", required=True, help="Output windows JSONL (one line per window).")
    ap.add_argument("--preview-out", default=None, help="Optional path to save a human-readable preview text.")
    ap.add_argument("--agent-id", type=int, default=None, help='Agent id under evaluation; selects actions.json key "0"/"1".')
    ap.add_argument("--human-id", type=int, default=None, help="Partner id; used to fetch personality context (best-effort).")
    ap.add_argument("--window-size", type=int, default=3, help="Window size (must be 3).")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of actions.json files processed (for testing).")
    ap.add_argument("--max-chars", type=int, default=2200, help="Truncate each field to this many chars to control window_text size.")
    ap.add_argument("--preview-max-chars", type=int, default=0, help="Truncate preview text; 0 means no truncation.")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    if data_root.name == "actions":
        data_root = data_root.parent
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text("", encoding="utf-8")

    if args.window_size != 3:
        raise SystemExit("This evaluator currently requires --window-size 3.")

    inferred_agent, inferred_human = _parse_agent_human_ids_from_root(data_root)
    agent_id = args.agent_id if args.agent_id is not None else inferred_agent
    human_id = args.human_id if args.human_id is not None else inferred_human
    if agent_id is None:
        raise SystemExit('Cannot infer agent id from data root; pass --agent-id (uses actions.json key "0"/"1").')

    eval_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(eval_dir))
    from llmjudge_format import format_window_for_judge, format_window_preview  # type: ignore
    from llmjudge_types import ActionWindow, WindowStep  # type: ignore

    preview_file = None
    if args.preview_out:
        preview_path = Path(args.preview_out).expanduser().resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text("", encoding="utf-8")
        preview_file = preview_path.open("w", encoding="utf-8")

    run_ts = _dt.datetime.now().isoformat(timespec="seconds")

    n_files = 0
    n_windows = 0

    for actions_path in _iter_actions_json(data_root, max_files=args.max_files):
        n_files += 1
        folder_name = actions_path.parents[1].name
        task_name, k, persona_id = _parse_task_and_persona(folder_name)

        raw = _read_json(actions_path)
        if not isinstance(raw, dict):
            continue
        agent_entries = raw.get(str(agent_id))
        if not isinstance(agent_entries, list) or not agent_entries:
            continue

        oppo_entries: List[Dict[str, Any]] = []
        if human_id is not None:
            oppo_raw = raw.get(str(human_id))
            if isinstance(oppo_raw, list):
                oppo_entries = [e for e in oppo_raw if isinstance(e, dict)]
        oppo_step_index = _build_step_index(oppo_entries) if oppo_entries else ([], [])

        num_full = len(agent_entries) // args.window_size
        if num_full <= 0:
            continue

        obj: Dict[str, Any] = {
            "source": "proagent_step1_actions_json",
            "data_root": str(data_root),
            "actions_json": str(actions_path),
            "actions_folder": folder_name,
            "task_name": task_name,
            "persona_id": persona_id,
            "window_size": args.window_size,
            "agent_high_level_calls": len(agent_entries),
            "agent_id": agent_id,
            "human_id": human_id,
            "k_prefix": k,
        }

        header = (
            f"# ProAgent windows: {actions_path} agent_id={agent_id} human_id={human_id} task={task_name} persona={persona_id}"
        )
        if preview_file:
            preview_file.write(header + "\n")

        for w in range(num_full):
            steps: List[WindowStep] = []
            llm_indices: List[int] = []
            step_nows: List[int] = []
            for j in range(args.window_size):
                idx = w * args.window_size + j
                e = agent_entries[idx]
                if not isinstance(e, dict):
                    e = {"raw": e}
                timestep = _safe_int(e.get("timestep"), idx)

                oppo_e = _latest_entry_at_or_before(oppo_step_index, timestep) if oppo_entries else None
                oppo_personality = str(oppo_e.get("personality", "")) if isinstance(oppo_e, dict) else ""

                agent_name = f"agent{agent_id}"
                oppo_name = f"human{human_id}" if human_id is not None else "human"
                scene_text = _dumps_compact(e.get("scene", ""))
                oppo_scene_text = _dumps_compact(oppo_e.get("scene", "")) if isinstance(oppo_e, dict) else ""

                conv_text = _to_dialogue_text(e.get("conversation_history"))
                action_text = _dumps_compact(e.get("action", ""))
                think_text = str(e.get("think", "")) if e.get("think") is not None else ""
                msg = e.get("message", None)
                msg_text = msg if isinstance(msg, str) else (None if msg is None else str(msg))
                outputs = e.get("outputs", None)
                outputs_text = outputs if isinstance(outputs, str) else (None if outputs is None else str(outputs))

                steps.append(
                    WindowStep(
                        llm_index=idx,
                        step_now=timestep,
                        agent_name=agent_name,
                        oppo_name=oppo_name,
                        personality=str(e.get("personality", "")),
                        oppo_personality=oppo_personality,
                        goal_desc="",
                        progress_desc=scene_text,
                        oppo_progress_desc=oppo_scene_text,
                        observation=scene_text,
                        oppo_observation=oppo_scene_text,
                        action_history_desc=action_text,
                        dialogue_history_desc=conv_text,
                        cot=think_text,
                        message=msg_text,
                        plan=action_text,
                        outputs=outputs_text,
                        outputs_source="outputs" if outputs_text else "",
                    )
                )
                llm_indices.append(idx)
                step_nows.append(timestep)

            window = ActionWindow(
                run_dir=str(actions_path.parents[2]),
                log_path=str(actions_path),
                task_id=-1,
                env_id=-1,
                task_name=task_name,
                agent_id=agent_id,
                window_index=w,
                steps=(steps[0], steps[1], steps[2]),
            )
            window_text = format_window_for_judge(window, args.max_chars)
            window_id = f"{folder_name}::step_1::agent{agent_id}::human{human_id}::win{w}"

            rec: Dict[str, Any] = {
                "run_ts": run_ts,
                "window_id": window_id,
                "run_dir": window.run_dir,
                "log_path": window.log_path,
                "task_id": window.task_id,
                "env_id": window.env_id,
                "task_name": window.task_name,
                "agent_id": agent_id,
                "human_id": human_id,
                "window_index": w,
                "llm_indices": llm_indices,
                "step_nows": step_nows,
                "agent_name": window.steps[0].agent_name,
                "human_name": window.steps[0].oppo_name,
                "window_text": window_text,
                "objective": obj,
            }
            _append_jsonl(out_path, rec)
            n_windows += 1

            if preview_file:
                preview_file.write(format_window_preview(window, max_chars=args.preview_max_chars) + "\n")

    if preview_file:
        preview_file.close()

    if n_files == 0:
        raise SystemExit(f"No step_1/actions.json found under: {data_root}")

    print(f"Wrote: {out_path} (files={n_files}, windows={n_windows})")
    return 0


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def cmd_run_pipeline(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run full llmjudge pipeline for ProAgent actions (step_1/actions.json).")
    ap.add_argument(
        "--data-root",
        required=True,
        help="ProAgent experiment root (e.g. .../src-0-agent-1-human-...); searches actions/**/step_1/actions.json",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: Evaluation/Judge/cook/out_<data_root_name>.",
    )
    ap.add_argument("--agent-id", type=int, default=None, help='Agent id under evaluation; selects actions.json key "0"/"1".')
    ap.add_argument("--human-id", type=int, default=None, help='Partner id; used for personality context (best-effort).')
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--b", type=float, default=0.8)
    ap.add_argument("--a-helpfulness", type=float, default=None)
    ap.add_argument("--b-helpfulness", type=float, default=None)
    ap.add_argument("--a-trustfulness", type=float, default=None)
    ap.add_argument("--b-trustfulness", type=float, default=None)
    ap.add_argument("--a-empathy", type=float, default=None)
    ap.add_argument("--b-empathy", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete large intermediate files after model_summary (windows/preview/requests/results_lite/results_detailed).",
    )
    ap.add_argument(
        "--no-proxy-host",
        default="localhost",
        help="Set NO_PROXY to this host to avoid local proxy TLS issues.",
    )

    args = ap.parse_args(argv)

    base = Path(__file__).resolve().parent
    eval_dir = base

    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()
    if data_root.name == "actions":
        data_root = data_root.parent
    if not data_root.exists():
        raise SystemExit(f"data-root not found: {data_root}")

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = (base / "cook" / f"out_{data_root.name}").resolve()
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

    env = dict(os.environ)
    env.setdefault("NO_PROXY", args.no_proxy_host)
    env.pop("HTTPS_PROXY", None)
    env.pop("HTTP_PROXY", None)
    env.pop("https_proxy", None)
    env.pop("http_proxy", None)

    Path(windows_jsonl).unlink(missing_ok=True)
    Path(windows_preview).unlink(missing_ok=True)
    Path(requests_jsonl).unlink(missing_ok=True)

    cmd = [
        "python",
        str(Path(__file__).resolve()),
        "build-windows",
        "--data-root",
        str(data_root),
        "--out",
        windows_jsonl,
        "--preview-out",
        windows_preview,
        "--window-size",
        "3",
        "--max-chars",
        str(args.max_chars),
    ]
    if args.max_files is not None:
        cmd += ["--max-files", str(args.max_files)]
    if args.agent_id is not None:
        cmd += ["--agent-id", str(args.agent_id)]
    if args.human_id is not None:
        cmd += ["--human-id", str(args.human_id)]
    _run(cmd, env=env)

    _run(
        [
            "python",
            str(eval_dir / "llmjudge.py"),
            "build-prompts",
            "--windows",
            windows_jsonl,
            "--out",
            requests_jsonl,
            "--thin",
        ],
        env=env,
    )

    cmd = [
        "python",
        str(eval_dir / "llmjudge.py"),
        "score-prompts",
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
        cmd.append("--resume")
    _run(cmd, env=env)

    _run(
        [
            "python",
            str(eval_dir / "llmjudge.py"),
            "backfill-objective",
            "--windows",
            windows_jsonl,
            "--results",
            results_lite,
            "--out",
            results_lite_obj,
            "--overwrite",
        ],
        env=env,
    )

    _run(
        [
            "python",
            str(eval_dir / "llmjudge.py"),
            "merge-outputs",
            "--windows",
            windows_jsonl,
            "--results",
            results_lite_obj,
            "--out-detailed",
            detailed,
            "--out-compact",
            compact,
        ],
        env=env,
    )

    _run(
        [
            "python",
            str(eval_dir / "llmjudge.py"),
            "export-window-csv",
            "--results",
            results_lite_obj,
            "--out-csv",
            window_csv,
        ],
        env=env,
    )

    cmd = [
        "python",
        str(eval_dir / "llmjudge.py"),
        "aggregate-results",
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
            cmd += [flag, str(val)]
    _run(cmd, env=env)

    _run(
        [
            "python",
            str(eval_dir / "llmjudge.py"),
            "model-summary",
            "--trajectory",
            traj_jsonl,
            "--out-json",
            model_summary_json,
            "--out-csv",
            model_summary_csv,
        ],
        env=env,
    )

    if args.cleanup:
        for p in [
            windows_jsonl,
            windows_preview,
            requests_jsonl,
            results_lite,
            detailed,
        ]:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    print(f"Done. outdir={outdir}")
    return 0


def _print_main_help() -> None:
    print("Usage: python cook.py <command> [args]")
    print("Commands:")
    print("  - build-windows")
    print("  - run-pipeline")
    print("Run: python cook.py <command> --help")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_main_help()
        return 0

    cmd = argv[0]
    args = argv[1:]
    commands = {
        "build-windows": cmd_build_windows,
        "run-pipeline": cmd_run_pipeline,
    }
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        _print_main_help()
        return 2

    return commands[cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
