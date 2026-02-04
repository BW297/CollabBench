# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional
from .base import BaseMemory
import sys
import os
import traceback
import ray
import csv
import random

# 添加导入路径
base_dir = os.path.dirname(__file__)
proagent_dir = os.path.join(base_dir, '..', 'environments', 'env_package', 'proagent')
lib_path = os.path.join(proagent_dir, 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
if proagent_dir not in sys.path:
    sys.path.insert(0, proagent_dir)

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

# 导入 Ray actor 和工具函数
from agent_system.environments.env_package.proagent.proagent.batch_client import get_global_batch_client
from agent_system.environments.env_package.proagent.proagent.agent_pair_worker import AgentPairWorker
from utils import NEW_LAYOUTS, OLD_LAYOUTS

import importlib_metadata
VERSION = importlib_metadata.version("overcooked_ai")


class proagent_agent_memory(BaseMemory):
    """
    ProAgent 环境的内存管理器：负责存储和获取每个环境的历史记录。
    管理 p0（强化学习训练）和 p1（ProAgent API 采样）两个智能体。
    """
    
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0
        self.p0_algo = 'RL'
        self.p1_algo = 'ProAgent'
        self.config = None
        self.agent_workers = []  # Ray AgentPair actors
        self.batch_client = None
        self.p1_infos = []
        self._persona_profiles: Optional[List[str]] = None
        self.persona_names: List[Optional[str]] = []

    @staticmethod
    def _normalize_sampling_parameters(sampling_parameters):
        if sampling_parameters is None:
            return {}
        if isinstance(sampling_parameters, dict):
            return dict(sampling_parameters)
        return {
            'prompt_level': getattr(sampling_parameters, 'prompt_level', 'l2-ap'),
            'belief_revision': getattr(sampling_parameters, 'belief_revision', False),
            'retrival_method': getattr(sampling_parameters, 'retrival_method', 'recent_k'),
            'K': getattr(sampling_parameters, 'K', 1),
            'using_big_5': getattr(sampling_parameters, 'using_big_5', False),
            'big_five': getattr(sampling_parameters, 'big_five', 'Extraversion'),
            'level': getattr(sampling_parameters, 'level', 'Low'),
        }

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def _load_persona_profiles(self) -> List[str]:
        """
        从 cook.csv 读取人格 Profile 列表（带缓存）。
        """
        if self._persona_profiles is not None:
            return self._persona_profiles

        try:
            # Path: coo/Running/Cook-MultiPlayer/src/prompts/cook.csv (relative to this file)
            base_dir = os.path.dirname(__file__)
            cook_csv_path = os.path.normpath(os.path.join(
                base_dir, "../../../../../Running/Cook-MultiPlayer/src/prompts/cook.csv"
            ))
            profiles = []
            with open(cook_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    profile = row.get("Profile", "").strip()
                    if profile:
                        profiles.append(profile)
            self._persona_profiles = profiles
        except Exception:
            self._persona_profiles = []
        return self._persona_profiles

    def reset(self, batch_size: int, obs, infos: List[Dict[str, Any]], 
              base_url: str = None, lm_id: str = None, prompt_template: str = None, 
              sampling_parameters: Dict[str, Any] = None, p0: str = 'RL', p1: str = 'ProAgent',
              batch_max_size: int = 8, batch_max_wait_time: float = 0.5, use_batch_client: bool = True,
              group_n: int = 1, seed: int = 0):
        """
        重置内存，初始化一批环境。
        
        参数:
            batch_size: 环境数量
            obs: 初始观察列表（状态列表）
            infos: 信息字典列表，包含 layout、mdp、horizon
            base_url: LLM API 的基础 URL（用于 p1 ProAgent）
            lm_id: LLM 模型 ID（用于 p1 ProAgent）
            prompt_template: 提示词模板（ProAgent 不使用，使用内部提示词）
            sampling_parameters: LLM 采样参数（用于 p1 ProAgent）
            p0: p0 使用的算法（默认为 'RL'）
            p1: p1 使用的算法（默认为 'ProAgent'）
        """
        if self._data is not None:
            self._data.clear()
        
        self.p1_algo = p1
        self.batch_size = batch_size
        self._data = None  # no local agents
        self.persona_names = [None] * batch_size

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # kill existing actors
        if self.agent_workers:
            for w in self.agent_workers:
                try:
                    ray.kill(w)
                except Exception:
                    pass
        self.agent_workers = []
        self.p1_infos = [{} for _ in range(batch_size)]

        # shared batch client for p1
        if p1 == 'ProAgent' and use_batch_client and base_url:
            self.batch_client = get_global_batch_client(
                base_url=base_url,
                lm_id=lm_id or 'gpt-3.5-turbo',
                max_batch_size=batch_max_size,
                max_wait_time=batch_max_wait_time,
                force_new=True,
            )
        else:
            self.batch_client = None

        # create per-env AgentPairWorker
        sp_norm = self._normalize_sampling_parameters(sampling_parameters)
        profiles = self._load_persona_profiles()
        rng = random.Random(seed)
        group_n = max(1, int(group_n))

        # 为每个逻辑环境（batch_size）分配独立的 profile
        # 每个逻辑环境的多个物理副本（group_n）将共享相同的 profile
        persona_by_env: List[Optional[str]] = []
        for _ in range(batch_size):
            persona_by_env.append(rng.choice(profiles) if profiles else None)

        for i in range(batch_size):
            layout = infos[i].get('layout', 'cramped_room')
            mdp = infos[i].get('mdp')
            if mdp is None:
                if VERSION == '1.1.0':
                    mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS.get(layout, layout))
                elif VERSION == '0.0.1':
                    mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS.get(layout, layout))
                else:
                    mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS.get(layout, layout))

            # 每个逻辑环境使用自己的 profile
            persona_name = persona_by_env[i] if persona_by_env else None
            worker = AgentPairWorker.remote(
                env_id=i,
                layout=layout,
                mdp=mdp,
                base_url=base_url,
                lm_id=lm_id or 'gpt-3.5-turbo',
                sampling_parameters=sp_norm,
                p0_algo=p0,
                p1_algo=p1,
                batch_client=self.batch_client,
                enable_batch_client=True,
                batch_max_size=batch_max_size,
                batch_max_wait_time=batch_max_wait_time,
                persona_name=persona_name,
            )
            self.agent_workers.append(worker)
            self.persona_names[i] = persona_name

        self.keys = None

    def build_text_obs(self, obs, goal=None):
        """
        调用 Ray actor 获取 p0 视角的文本观察
        """
        futs = []
        for i, worker in enumerate(self.agent_workers):
            state = obs[i] if isinstance(obs, list) else obs
            futs.append(worker.build_text_obs.remote(state))
        return ray.get(futs)
    
    def build_text_state_obs(self, obs, goal=None):
        """
        调用 Ray actor 获取 p0 视角的文本观察
        """
        futs = []
        for i, worker in enumerate(self.agent_workers):
            state = obs[i] if isinstance(obs, list) else obs
            futs.append(worker.build_text_state_obs.remote(state))
        return ray.get(futs)

    def build_anchor_obs(self, obs, goal=None):
        futs = []
        for i, worker in enumerate(self.agent_workers):
            state = obs[i] if isinstance(obs, list) else obs
            futs.append(worker.build_anchor_obs.remote(state))
        return ray.get(futs)

    def get_actions(self, actions, obs, goal, action_infos, cnt, ml_actions_for_p0=None):
        """
        获取两个玩家的动作。
        
        参数:
            actions: RL 智能体的文本动作列表（用于 p0）
            obs: 当前状态列表
            goal: Overcooked 不使用
            action_infos: 之前的动作信息
            cnt: 步数计数器
            ml_actions_for_p0: p0 的 ML 动作字符串列表（高阶动作），可选
        
        返回:
            dict_action_list: 字典列表 {0: action_p0, 1: action_p1}
            dict_info_list: 动作信息列表
        """
        dict_action_list = []
        dict_info_list = []

        futures = []
        for i, worker in enumerate(self.agent_workers):
            state = obs[i] if isinstance(obs, list) else obs
            text_action = actions[i] if isinstance(actions, list) else actions
            prev_info0 = {}
            if isinstance(action_infos, list) and i < len(action_infos) and isinstance(action_infos[i], dict):
                prev_info0 = action_infos[i].get(0, {}) or {}
            futures.append(worker.act.remote(text_action, state, prev_info0, cnt))

        results = ray.get(futures)
        for res in results:
            actions_dict = res.get("actions", {0: Action.STAY, 1: Action.STAY})
            infos_dict = res.get("infos", {0: {}, 1: {}})
            state_updates = res.get("state_updates", {})
            if isinstance(infos_dict.get(1), dict):
                infos_dict[1]["state_updates"] = state_updates
            dict_action_list.append(actions_dict)
            # print("---",actions_dict,"---")
            dict_info_list.append(infos_dict)

        return dict_action_list, dict_info_list

    def _text_to_action(self, text_action: str):
        """将文本动作转换为 Action 对象"""
        if text_action is None:
            return Action.STAY
        
        # 尝试从文本解析动作
        text_action = text_action.upper().strip()
        
        # 映射常见的动作字符串
        # 注意：方向（NORTH, SOUTH, EAST, WEST）是 Direction 类的属性
        # Action 类只有 STAY 和 INTERACT
        action_map = {
            'NORTH': Direction.NORTH,
            'SOUTH': Direction.SOUTH,
            'EAST': Direction.EAST,
            'WEST': Direction.WEST,
            'INTERACT': Action.INTERACT,
            'STAY': Action.STAY,
            'UP': Direction.NORTH,
            'DOWN': Direction.SOUTH,
            'LEFT': Direction.WEST,
            'RIGHT': Direction.EAST,
        }
        
        if text_action in action_map:
            return action_map[text_action]
        
        # 尝试从动作字符串中提取
        for action_name, action_obj in action_map.items():
            if action_name in text_action:
                return action_obj
        
        # 默认返回 STAY
        return Action.STAY

    def _state_to_text(self, state):
        """将状态转换为文本表示"""
        if state is None:
            return "Unknown state"
        
        # 使用状态的字符串表示
        try:
            if hasattr(state, 'mdp') and hasattr(state.mdp, 'state_string'):
                return state.mdp.state_string(state).replace('ø', 'o')
            else:
                return str(state)
        except:
            return str(state)

    def _state_to_anchor(self, state, player_idx=0, ml_action=None):
        """将状态转换为锚点观察字典"""
        if state is None:
            return {}
        
        anchor = {}
        try:
            if hasattr(state, 'players') and player_idx < len(state.players):
                player = state.players[player_idx]
                anchor['position'] = player.position
                anchor['orientation'] = str(player.orientation)
                anchor['held_object'] = player.held_object.name if player.held_object else 'nothing'
            
            if ml_action is not None:
                anchor['ml_action'] = ml_action
        except:
            pass
        
        return anchor

    def get_persona_name(self, idx: int) -> Optional[str]:
        """
        获取指定环境的 persona 名称，如果不存在则返回 None。
        """
        if self.persona_names is None:
            return None
        if idx < 0 or idx >= len(self.persona_names):
            return None
        return self.persona_names[idx]

    def get_p0_profile(self, env_idx: int) -> Optional[str]:
        """获取指定环境 p0 agent 的 profile（即 persona_name）"""
        return self.get_persona_name(env_idx)

    def store(self, record: Dict[str, List[Any]]):
        """
        将新的一批记录存储到内存中。
        """
        pass

    def fetch(self, step: int):
        """
        获取特定时间步的所有环境的内存记录。
        """
        pass
    
    def close(self):
        """Close Ray actors managed by this memory"""
        if self.agent_workers:
            for worker in self.agent_workers:
                if worker is not None:
                    try:
                        ray.kill(worker)
                    except Exception as e:
                        print(f"Warning: Failed to kill agent worker: {e}")
            self.agent_workers = []
        if self.batch_client is not None:
            try:
                ray.kill(self.batch_client)
            except Exception as e:
                print(f"Warning: Failed to kill batch client: {e}")
            self.batch_client = None

    def check_ml_action_done(self, env_idx: int, state: Any) -> bool:
        """
        检查指定环境的 agent0 ML action 是否完成
        
        参数:
            env_idx: 环境索引
            state: 当前状态
        
        返回:
            bool: ML action 是否完成
        """
        if env_idx >= len(self.agent_workers):
            return True
        
        try:
            worker = self.agent_workers[env_idx]
            return ray.get(worker.check_ml_action_done.remote(state))
        except Exception:
            return True