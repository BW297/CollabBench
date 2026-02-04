from __future__ import annotations

import bisect
import json
from typing import Any, Dict, List, Optional, Tuple

from llmjudge_types import ActionWindow, WindowStep
from llmjudge_utils import safe_int


def _build_step_index(entries: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
    pairs: List[Tuple[int, Dict[str, Any]]] = []
    for e in entries:
        try:
            s = int(e.get("step_now"))
        except Exception:
            continue
        if s >= 0:
            pairs.append((s, e))
    pairs.sort(key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _latest_entry_at_or_before(
    step_index: Tuple[List[int], List[Dict[str, Any]]], step_now: int
) -> Optional[Dict[str, Any]]:
    steps, vals = step_index
    if not steps:
        return None
    if step_now < 0:
        return vals[-1]
    pos = bisect.bisect_right(steps, step_now) - 1
    if pos < 0:
        return vals[0]
    return vals[pos]


def _obs_to_text(obs_series: Any, step_now: int) -> str:
    if not isinstance(obs_series, list):
        return ""
    if step_now < 0 or step_now >= len(obs_series):
        return ""
    o = obs_series[step_now]
    if not isinstance(o, dict):
        return str(o)
    keep = {}
    for k in ["current_room", "grabbed_objects", "opponent_grabbed_objects", "reachable_objects", "progress", "satisfied"]:
        if k in o:
            keep[k] = o[k]
    try:
        return json.dumps(keep, ensure_ascii=False)
    except Exception:
        return str(keep)


def build_windows_from_log(
    log: Dict[str, Any],
    *,
    run_dir: str,
    log_path: str,
    agent_id: int,
    oppo_id: Optional[int],
    fallback_oppo_personality: Optional[str] = None,
    window_size: int,
) -> List[ActionWindow]:
    if window_size != 3:
        raise ValueError("This evaluator currently requires window_size=3.")

    llm_entries: List[Dict[str, Any]] = log.get("LLM", {}).get(agent_id, [])
    if not llm_entries:
        return []

    task_id = safe_int(log.get("task_id"), -1)
    env_id = safe_int(log.get("env_id"), -1)
    task_name = str(log.get("task_name", ""))

    obs_series_agent = log.get("obs", {}).get(agent_id) if isinstance(log.get("obs"), dict) else None
    obs_series_oppo = (
        log.get("obs", {}).get(oppo_id) if (oppo_id is not None and isinstance(log.get("obs"), dict)) else None
    )

    oppo_entries: List[Dict[str, Any]] = []
    if oppo_id is not None:
        oppo_entries = log.get("LLM", {}).get(oppo_id, []) or []
    oppo_step_index = _build_step_index(oppo_entries) if oppo_entries else ([], [])

    windows: List[ActionWindow] = []
    num_full = len(llm_entries) // window_size
    for w in range(num_full):
        steps: List[WindowStep] = []
        for j in range(window_size):
            idx = w * window_size + j
            e = llm_entries[idx]
            step_now = safe_int(e.get("step_now"), -1)

            oppo_e = _latest_entry_at_or_before(oppo_step_index, step_now) if oppo_entries else None
            oppo_personality = str(oppo_e.get("personality", "")) if isinstance(oppo_e, dict) else ""
            if (not oppo_personality.strip()) and fallback_oppo_personality:
                oppo_personality = fallback_oppo_personality
            oppo_progress_desc = str(oppo_e.get("progress_desc", "")) if isinstance(oppo_e, dict) else ""

            msg = e.get("message", None)
            msg_text = msg if isinstance(msg, str) else (None if msg is None else str(msg))

            outputs_source = ""
            outputs_text: Optional[str] = None
            if not (msg_text or "").strip():
                if isinstance(e.get("raw_output"), str) and e["raw_output"].strip():
                    outputs_source = "raw_output"
                    outputs_text = e["raw_output"]
                elif isinstance(e.get("outputs"), str) and e["outputs"].strip():
                    outputs_source = "outputs"
                    outputs_text = e["outputs"]
                elif isinstance(e.get("raw_outputs"), list) and e["raw_outputs"]:
                    outputs_source = "raw_outputs"
                    outputs_text = "\n".join(str(x) for x in e["raw_outputs"] if x is not None)

            steps.append(
                WindowStep(
                    llm_index=idx,
                    step_now=step_now,
                    agent_name=str(e.get("agent_name", "")),
                    oppo_name=str(e.get("oppo_name", "")),
                    personality=str(e.get("personality", "")),
                    oppo_personality=oppo_personality,
                    goal_desc=str(e.get("goal_desc", "")),
                    progress_desc=str(e.get("progress_desc", "")),
                    oppo_progress_desc=oppo_progress_desc,
                    observation=_obs_to_text(obs_series_agent, step_now),
                    oppo_observation=_obs_to_text(obs_series_oppo, step_now),
                    action_history_desc=str(e.get("action_history_desc", "")),
                    dialogue_history_desc=str(e.get("dialogue_history_desc", "")),
                    cot=str(e.get("think", "")),
                    message=msg_text,
                    plan=str(e.get("plan", "")),
                    outputs=outputs_text,
                    outputs_source=outputs_source,
                )
            )

        windows.append(
            ActionWindow(
                run_dir=run_dir,
                log_path=log_path,
                task_id=task_id,
                env_id=env_id,
                task_name=task_name,
                agent_id=agent_id,
                window_index=w,
                steps=(steps[0], steps[1], steps[2]),
            )
        )
    return windows
