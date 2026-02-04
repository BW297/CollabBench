from __future__ import annotations

import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass(frozen=True)
class WindowStep:
    llm_index: int
    step_now: int
    agent_name: str
    oppo_name: str
    personality: str
    oppo_personality: str
    goal_desc: str
    progress_desc: str
    oppo_progress_desc: str
    observation: str
    oppo_observation: str
    action_history_desc: str
    dialogue_history_desc: str
    cot: str
    message: Optional[str]
    plan: str
    outputs: Optional[str]
    outputs_source: str


@dataclasses.dataclass(frozen=True)
class ActionWindow:
    run_dir: str
    log_path: str
    task_id: int
    env_id: int
    task_name: str
    agent_id: int
    window_index: int
    steps: Tuple[WindowStep, WindowStep, WindowStep]

