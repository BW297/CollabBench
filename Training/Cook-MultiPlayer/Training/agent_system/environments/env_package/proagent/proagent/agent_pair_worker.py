import copy
from typing import Any, Dict, Optional
import ray

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner

from proagent.proagent import ProMediumLevelAgent
from .batch_client import get_global_batch_client


@ray.remote
class AgentPairWorker:
    """
    Ray actor owning both agents (p0 RL-side, p1 API-side).
    p0: 解析/执行 RL 的文本高阶动作 (ML action)
    p1: ProAgent，使用共享的异步批处理 LLM 调度器
    """

    def __init__(
        self,
        env_id: int,
        layout: str,
        mdp: OvercookedGridworld,
        base_url: Optional[str],
        lm_id: str,
        sampling_parameters: Dict[str, Any],
        p0_algo: str = "RL",
        p1_algo: str = "ProAgent",
        batch_client: Optional[ray.actor.ActorHandle] = None,
        enable_batch_client: bool = True,
        batch_max_size: int = 8,
        batch_max_wait_time: float = 0.5,
        persona_name: Optional[str] = None,
    ):
        self.env_id = env_id
        self.layout = layout
        self.base_url = base_url
        self.lm_id = lm_id or "gpt-3.5-turbo"
        self.p0_algo = p0_algo
        self.p1_algo = p1_algo

        # normalize params
        sp = sampling_parameters or {}
        self.prompt_level = sp.get("prompt_level", "l2-ap")
        self.belief_revision = sp.get("belief_revision", False)
        self.retrival_method = sp.get("retrival_method", "recent_k")
        self.K = sp.get("K", 1)
        self.using_big_5 = sp.get("using_big_5", False)
        self.big_5 = sp.get("big_five", "Extraversion")
        self.level = sp.get("level", "Low")

        ml_params = {
            "start_orientations": False,
            "wait_allowed": True,
            "counter_goals": mdp.get_counter_locations(),
            "counter_drop": mdp.get_counter_locations(),
            "counter_pickup": mdp.get_counter_locations(),
            "same_motion_goals": True,
        }
        self.mlam = MediumLevelPlanner.from_pickle_or_compute(
            mdp, ml_params, force_compute=True
        ).ml_action_manager

        # batch client (shared)
        self.batch_client = batch_client
        if p1_algo == "ProAgent" and enable_batch_client and base_url and batch_client is None:
            self.batch_client = get_global_batch_client(
                base_url=base_url,
                lm_id=self.lm_id,
                max_batch_size=batch_max_size,
                max_wait_time=batch_max_wait_time,
                force_new=False,
            )

        # p0
        self.agent0 = None
        if p0_algo == "RL":
            self.agent0 = ProMediumLevelAgent(
                self.mlam,
                layout,
                model=self.lm_id,
                prompt_level=self.prompt_level,
                belief_revision=self.belief_revision,
                retrival_method=self.retrival_method,
                K=self.K,
                using_big_5=self.using_big_5,
                big_5=self.big_5,
                level=self.level,
                base_url=base_url,
                api_key="EMPTY",
                worker_id=env_id,
                profile=persona_name,
            )
            self.agent0.set_agent_index(0, batch_client=None)
            self.agent0.set_mdp(mdp)
            self.agent0.reset()

        # p1
        self.agent1 = None
        if p1_algo == "ProAgent":
            self.agent1 = ProMediumLevelAgent(
                self.mlam,
                layout,
                model=self.lm_id,
                prompt_level=self.prompt_level,
                belief_revision=self.belief_revision,
                retrival_method=self.retrival_method,
                K=self.K,
                using_big_5=self.using_big_5,
                big_5=self.big_5,
                level=self.level,
                base_url=base_url,
                api_key="EMPTY",
                worker_id=env_id,
                profile=persona_name,
            )
            self.agent1.set_agent_index(1, batch_client=self.batch_client)
            self.agent1.set_mdp(mdp)
            self.agent1.reset()

    # ---- helper: capture mutated state for p1 ----
    def _capture_state_updates(self, state: Any) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        for attr in [
            "behavior_list",
            "scene_list",
            "chat_list",
            "fail_back",
            "state_back",
            "haschat",
            "haschat_other",
            "ml_actions",
            "steps",
        ]:
            if hasattr(state, attr):
                updates[attr] = copy.deepcopy(getattr(state, attr))
        return updates

    # ---- main API ----
    def act(
        self,
        text_action: str,
        state: Any,
        action_info_p0: Optional[Dict] = None,
        cnt: int = 0,
    ):
        """
        text_action: RL 文本高阶动作
        state: env.state
        action_info_p0: 上一步的 info_p0（用于 ml_action_completed 等标记）
        """
        if action_info_p0 is None:
            action_info_p0 = {}

        # p0
        dict_info0 = {}
        action0 = Action.STAY
        if self.agent0 is not None:
            try:
                if cnt == 0:
                    parsed_ml = self.agent0.parse_ml_action(text_action, agent_index=0)
                    # print(f"parsed_ml: {parsed_ml}")
                    if not hasattr(self.agent0, "_external_ml_action"):
                        self.agent0._external_ml_action = None
                    self.agent0._external_ml_action = parsed_ml
                    print("parsed_ml", parsed_ml)
                    # ensure state fields
                    for attr, init_v in [
                        ("behavior_list", {"P0": [], "P1": []}),
                        ("scene_list", {"P0": [], "P1": []}),
                        ("state_back", {"P0": [], "P1_done": [], "P1": []}),
                        ("fail_back", {"P0": [], "P1": []}),
                    ]:
                        if not hasattr(state, attr) or getattr(state, attr) is None:
                            setattr(state, attr, copy.deepcopy(init_v))
                    if len(state.behavior_list.get("P0", [])) == 0:
                        state.behavior_list["P0"] = [{"Plan": parsed_ml}]
                    if len(state.scene_list.get("P0", [])) == 0:
                        state.scene_list["P0"] = [""]
                    if len(state.state_back.get("P0", [])) == 0:
                        state.state_back["P0"] = [{}]

                    self.agent0.current_ml_action = parsed_ml
                    self.agent0.current_ml_action_steps = 0

                if action_info_p0.get("ml_action_completed", False):
                    action0 = Action.STAY
                    dict_info0 = action_info_p0
                else:
                    action0, info0 = self.agent0.action(state)
                    dict_info0 = info0 or {}
                    if hasattr(self.agent0, "_external_ml_action") and self.agent0._external_ml_action is not None:
                        dict_info0["ml_action"] = self.agent0._external_ml_action
                    else:
                        dict_info0["ml_action"] = getattr(self.agent0, "current_ml_action", None)
                    dict_info0.setdefault("source", "RL")
                    dict_info0["text_action"] = text_action
            except Exception as e:
                dict_info0 = {"source": "RL", "error": str(e)}
                action0 = Action.STAY

        # p1
        dict_info1 = {}
        action1 = Action.STAY
        state_updates = {}
        if self.agent1 is not None:
            try:
                action1, info1 = self.agent1.action(state)
                dict_info1 = info1 or {}
                dict_info1.setdefault("ml_action", getattr(self.agent1, "current_ml_action", None))
                dict_info1.setdefault("source", "ProAgent")
                state_updates = self._capture_state_updates(state)
            except Exception as e:
                dict_info1 = {"source": "ProAgent", "error": str(e)}
                action1 = Action.STAY
                state_updates = {}

        return {
            "actions": {0: action0, 1: action1},
            "infos": {0: dict_info0, 1: dict_info1},
            "state_updates": state_updates,
        }

    # ---- observation helpers ----
    def build_text_obs(self, state: Any):
        if self.agent0 is None:
            return "Unknown state"
        try:
            if not hasattr(state, "behavior_list"):
                state.behavior_list = {"P0": [], "P1": []}
            if not hasattr(state, "scene_list"):
                state.scene_list = {"P0": [], "P1": []}
            if not hasattr(state, "haschat"):
                state.haschat = {"P0": False, "P1": False}
            if not hasattr(state, "chat_list"):
                state.chat_list = {"P0": [], "P1": []}

            if self.agent0.prompt_level == "l2-ap_merged" and hasattr(
                self.agent0, "_update_planner_prompt_for_state"
            ):
                self.agent0._update_planner_prompt_for_state(state)

            belief_prompt = ""
            if self.agent0.prompt_level == "l3-aip" and self.agent0.belief_revision:
                belief_prompt = self.agent0.generate_belief_prompt()

            state_prompt = (
                belief_prompt
                + self.agent0.generate_trace_prompt(state.behavior_list, state.scene_list)
                + "\n\n"
                + self.agent0.generate_state_prompt(state)
                + "\n\n"
                + self.agent0.generate_team_chat_prompt(state.haschat, state.chat_list, 1)
            )

            parts = []
            if hasattr(self.agent0, "planner") and self.agent0.planner.instruction_head_list:
                system_prompt = self.agent0.planner.instruction_head_list[0].get("content", "")
                if system_prompt:
                    parts.append(f"System: {system_prompt}")
            if hasattr(self.agent0, "planner"):
                cache_list = self.agent0.planner.get_cache()
                if cache_list:
                    for msg in cache_list:
                        parts.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
            parts.append(f"User: {state_prompt}")

            # print(f"parts: {parts}")
            return "\n\n".join(parts)
        except Exception:
            return str(state)
        
    def build_text_state_obs(self, state: Any):

        state_prompt = (
            self.agent0.generate_trace_prompt(state.behavior_list, state.scene_list)
            + "\n\n"
            + self.agent0.generate_state_prompt(state)
            + "\n\n"
            + self.agent0.generate_team_chat_prompt(state.haschat, state.chat_list, 1)
        )
        return state_prompt


    def build_anchor_obs(self, state: Any):
        anchor0 = {}
        kitchen_layout = ""
        if self.agent0 and hasattr(self.agent0, 'layout_prompt'):
            kitchen_layout = self.agent0.layout_prompt

        try:
            if hasattr(state, "players"):
                p0 = state.players[0]
                p1 = state.players[1]
                # anchor0["layout"] = kitchen_layout
                anchor0["teammate_held_object"] = p1.held_object.name if p1.held_object else "nothing"
                anchor0["state"] = self.agent0.generate_kitchen_prompt(state)
                anchor0["held_object"] = p0.held_object.name if p0.held_object else "nothing"
            anchor0["layout"] =kitchen_layout
        except Exception:
            pass
        anchor1 = {"ml_action": getattr(self.agent1, "current_ml_action", None) if self.agent1 else None}
        return {0: anchor0, 1: anchor1}

    def check_ml_action_done(self, state: Any) -> bool:
        """
        检查 agent0 的 ML action 是否完成
        
        参数:
            state: 当前环境状态
        
        返回:
            bool: ML action 是否完成
        """
        if self.agent0 is None:
            return True
        
        try:
            # 检查是否有外部 ML 动作
            if not hasattr(self.agent0, '_external_ml_action') or self.agent0._external_ml_action is None:
                return True
            
            # 调用 agent 的检查方法
            if hasattr(self.agent0, 'check_current_ml_action_done'):
                return self.agent0.check_current_ml_action_done(state)
            
            return False
        except Exception:
            return False
