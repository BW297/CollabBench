from typing import Any, Dict, List, Optional
import ray
import copy

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from proagent.proagent import ProMediumLevelAgent

from .batch_client import ProAgentBatchClientWorker


@ray.remote
class P1ProAgentWorker:
    def __init__(
        self,
        env_id: int,
        layout: str,
        mdp: OvercookedGridworld,
        base_url: Optional[str],
        lm_id: str,
        sampling_parameters: Dict[str, Any],
        batch_client: Optional[ray.actor.ActorHandle] = None,
        persona_name: Optional[str] = None,
    ):
        self.env_id = env_id
        self.layout = layout
        self.base_url = base_url
        self.lm_id = lm_id
        self.batch_client = batch_client

        self.prompt_level = sampling_parameters.get("prompt_level", "l2-ap")
        self.belief_revision = sampling_parameters.get("belief_revision", False)
        self.retrival_method = sampling_parameters.get("retrival_method", "recent_k")
        self.K = sampling_parameters.get("K", 1)
        self.using_big_5 = sampling_parameters.get("using_big_5", False)
        self.big_5 = sampling_parameters.get("big_five", "Extraversion")
        self.level = sampling_parameters.get("level", "Low")

        ml_params = {
            "start_orientations": False,
            "wait_allowed": True,
            "counter_goals": mdp.get_counter_locations(),
            "counter_drop": mdp.get_counter_locations(),
            "counter_pickup": mdp.get_counter_locations(),
            "same_motion_goals": True,
        }

        self.mlam = MediumLevelPlanner.from_pickle_or_compute(mdp, ml_params, force_compute=True).ml_action_manager

        self.agent = ProMediumLevelAgent(
            self.mlam,
            layout,
            model=lm_id,
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
        self.agent.set_agent_index(1, batch_client=batch_client)
        self.agent.set_mdp(mdp)

    def act(self, state: Any) -> Dict[str, Any]:
        """
        Execute ProAgent action for the provided state and return action/info/state updates.
        """
        action_p1, info_p1 = self.agent.action(state)

        if info_p1 is None:
            info_p1 = {}

        state_updates = self._capture_state_updates(state)

        info_p1.setdefault("ml_action", getattr(self.agent, "current_ml_action", None))
        info_p1.setdefault("source", "ProAgent")

        return {
            "action": action_p1,
            "info": info_p1,
            "state_updates": state_updates,
        }

    def _capture_state_updates(self, state: Any) -> Dict[str, Any]:
        """
        Capture the state fragments that ProAgent mutated so they can be synced back
        to the env worker.
        """
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
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import ray
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner

from .proagent import ProMediumLevelAgent

MLAM_PARAMS = {
    "start_orientations": False,
    "wait_allowed": True,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}


def _parse_sampling_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if params is None:
        params = {}
    prompt_level = params.get("prompt_level", "l2-ap")
    belief_revision = params.get("belief_revision", False)
    retrival_method = params.get("retrival_method", "recent_k")
    K = params.get("K", 1)
    using_big_5 = params.get("using_big_5", False)
    big_5 = params.get("big_five", "Extraversion")
    level = params.get("level", "Low")
    return {
        "prompt_level": prompt_level,
        "belief_revision": belief_revision,
        "retrival_method": retrival_method,
        "K": K,
        "using_big_5": using_big_5,
        "big_5": big_5,
        "level": level,
    }


@ray.remote
class P1ProAgentWorker:
    def __init__(
        self,
        env_id: int,
        layout: str,
        mdp: OvercookedGridworld,
        base_url: Optional[str],
        lm_id: Optional[str],
        sampling_parameters: Optional[Dict[str, Any]],
        batch_client: Optional[ray.actor.ActorHandle] = None,
        persona_name: Optional[str] = None,
    ):
        self.env_id = env_id
        self.layout = layout
        self.base_url = base_url
        self.lm_id = lm_id or "gpt-3.5-turbo"
        self.batch_client = batch_client
        self.mdp = mdp
        self.sampling_parameters = _parse_sampling_params(sampling_parameters)
        self.persona_name = persona_name
        self.agent: Optional[ProMediumLevelAgent] = None
        self._build_agent()

    def _build_agent(self):
        counter_locations = self.mdp.get_counter_locations()
        params = MLAM_PARAMS.copy()
        params["counter_goals"] = counter_locations
        params["counter_drop"] = counter_locations
        params["counter_pickup"] = counter_locations
        mlam = MediumLevelPlanner.from_pickle_or_compute(self.mdp, params, force_compute=True).ml_action_manager
        self.agent = ProMediumLevelAgent(
            mlam,
            self.layout,
            model=self.lm_id,
            prompt_level=self.sampling_parameters["prompt_level"],
            belief_revision=self.sampling_parameters["belief_revision"],
            retrival_method=self.sampling_parameters["retrival_method"],
            K=self.sampling_parameters["K"],
            using_big_5=self.sampling_parameters["using_big_5"],
            big_5=self.sampling_parameters["big_5"],
            level=self.sampling_parameters["level"],
            base_url=self.base_url,
            api_key="EMPTY",
            worker_id=self.env_id,
            profile=self.persona_name,
        )
        self.agent.set_agent_index(1, batch_client=self.batch_client)
        self.agent.set_mdp(self.mdp)
        self.agent.reset()

    def _capture_state_updates(self, state: Any) -> Dict[str, Any]:
        fields = [
            "behavior_list",
            "scene_list",
            "chat_list",
            "fail_back",
            "state_back",
            "haschat",
            "haschat_other",
            "ml_actions",
            "steps",
        ]
        updates: Dict[str, Any] = {}
        for field in fields:
            value = getattr(state, field, None)
            if value is not None:
                updates[field] = copy.deepcopy(value)
        return updates

    def act(self, state: Any) -> Dict[str, Any]:
        if self.agent is None:
            return {"action": Action.STAY, "info": {}, "state_updates": {}}
        try:
            action_p1, info_p1 = self.agent.action(state)
            if info_p1 is None:
                info_p1 = {}
            if "ml_action" not in info_p1:
                info_p1["ml_action"] = getattr(self.agent, "current_ml_action", None)
            info_p1.setdefault("source", "ProAgent")
        except Exception as exc:
            info_p1 = {"source": "ProAgent", "error": str(exc)}
            action_p1 = Action.STAY
        updates = self._capture_state_updates(state)
        return {"action": action_p1, "info": info_p1, "state_updates": updates}

    def reset(self) -> None:
        if self.agent is not None:
            self.agent.reset()
