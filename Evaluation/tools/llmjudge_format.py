from __future__ import annotations

from llmjudge_types import ActionWindow
from llmjudge_utils import plan_to_high_level_action, truncate


def format_window_for_judge(window: ActionWindow, max_chars_per_field: int) -> str:
    header = [
        f"run_dir: {window.run_dir}",
        f"log_path: {window.log_path}",
        f"task_id: {window.task_id}  env_id: {window.env_id}  task_name: {window.task_name}",
        f"agent_id: {window.agent_id}  window_index: {window.window_index}",
        "",
        "=== WINDOW (3 decision steps) ===",
    ]
    body: list[str] = []
    for i, st in enumerate(window.steps, start=1):
        use_outputs = (not (st.message or "").strip()) and bool((st.outputs or "").strip())
        body.extend(
            [
                f"[Step {i}] llm_index={st.llm_index} step_now={st.step_now} agent={st.agent_name} oppo={st.oppo_name}",
                # NOTE: In your setting, personality is only defined for human (partner/oppo).
                "human_personality: " + truncate(st.oppo_personality, max_chars_per_field),
                "目标(goal): " + truncate(st.goal_desc, max_chars_per_field),
                "观测(progress): " + truncate(st.progress_desc, max_chars_per_field),
                "对方观测(oppo_progress): " + truncate(st.oppo_progress_desc, max_chars_per_field),
                "观测(observation_raw): " + truncate(st.observation, max_chars_per_field),
                "对方观测(oppo_observation_raw): " + truncate(st.oppo_observation, max_chars_per_field),
                "动作历史(action_history): " + truncate(st.action_history_desc, max_chars_per_field),
                "对话历史(dialogue_history): " + truncate(st.dialogue_history_desc, max_chars_per_field),
                (
                    "模型原始输出(raw_output): "
                    + truncate(st.outputs or "", max_chars_per_field)
                    + (f" (source={st.outputs_source})" if st.outputs_source else "")
                )
                if use_outputs
                else "思维链(cot): " + truncate(st.cot, max_chars_per_field),
                "" if use_outputs else "message: " + truncate(st.message or "", max_chars_per_field),
                "" if use_outputs else "高阶动作(action/plan): " + truncate(plan_to_high_level_action(st.plan), max_chars_per_field),
                "",
            ]
        )
    return "\n".join(header + body).strip() + "\n"


def format_window_preview(window: ActionWindow, *, max_chars: int) -> str:
    lines: list[str] = [f"== window {window.window_index} (agent_id={window.agent_id}) =="]
    for i, st in enumerate(window.steps, start=1):
        use_outputs = (not (st.message or "").strip()) and bool((st.outputs or "").strip())
        lines.append(f"[{i}] llm_index={st.llm_index} step_now={st.step_now} agent={st.agent_name} oppo={st.oppo_name}")
        if st.oppo_personality:
            lines.append(f"    human_personality: {truncate(st.oppo_personality, max_chars)}")
        if st.oppo_progress_desc:
            lines.append(f"    oppo_progress: {truncate(st.oppo_progress_desc, max_chars)}")
        if st.observation:
            lines.append(f"    observation_raw: {truncate(st.observation, max_chars)}")
        if st.oppo_observation:
            lines.append(f"    oppo_observation_raw: {truncate(st.oppo_observation, max_chars)}")
        if use_outputs:
            lines.append(f"    raw_output({st.outputs_source}): {truncate(st.outputs, max_chars)}")
        else:
            lines.append(f"    think: {truncate(st.cot, max_chars)}")
            lines.append(f"    plan: {plan_to_high_level_action(st.plan)}")
            if st.message:
                lines.append(f"    message: {truncate(st.message, max_chars)}")
        if st.action_history_desc:
            lines.append(f"    action_history: {truncate(st.action_history_desc, max_chars)}")
        if st.progress_desc:
            lines.append(f"    progress: {truncate(st.progress_desc, max_chars)}")
    return "\n".join(lines)
