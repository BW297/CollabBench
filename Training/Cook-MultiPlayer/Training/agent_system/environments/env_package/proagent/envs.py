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

import random
import gymnasium as gym
import numpy as np
import ray
import sys
import os
import traceback

# Add lib to path for overcooked_ai import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    BASE_REW_SHAPING_PARAMS,
    NO_REW_SHAPING_PARAMS,
)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction

# Import utils for layout mapping
sys.path.insert(0, os.path.dirname(__file__))
from utils import NEW_LAYOUTS, OLD_LAYOUTS, ALL_LAYOUTS, get_layout_for_env

import importlib_metadata
VERSION = importlib_metadata.version("overcooked_ai")

def env_fn(seed, env_id, base_port, is_train, env_kwargs, layout_override=None):
    """Create a single Overcooked environment instance
    
    Args:
        seed: Random seed
        env_id: Environment index
        base_port: Base port for communication
        is_train: Whether this is a training environment
        env_kwargs: Environment keyword arguments
        layout_override: If provided, use this layout instead of env_kwargs['layout']
    """
    layout = layout_override if layout_override is not None else env_kwargs.get('layout', 'cramped_room')
    horizon = env_kwargs.get('horizon', 400)

    # Reward shaping configuration
    enable_reward_shaping = env_kwargs.get("enable_reward_shaping", True)
    rew_shaping_params = env_kwargs.get("rew_shaping_params", None)
    if rew_shaping_params is None:
        rew_shaping_params = BASE_REW_SHAPING_PARAMS if enable_reward_shaping else NO_REW_SHAPING_PARAMS
    
    # Create MDP based on version
    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(
            NEW_LAYOUTS.get(layout, layout),
            rew_shaping_params=rew_shaping_params
        )
    elif VERSION == '0.0.1':
        mdp = OvercookedGridworld.from_layout_name(
            OLD_LAYOUTS.get(layout, layout),
            rew_shaping_params=rew_shaping_params
        )
    else:
        mdp = OvercookedGridworld.from_layout_name(
            NEW_LAYOUTS.get(layout, layout),
            rew_shaping_params=rew_shaping_params
        )
    
    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()
    
    # Initialize state attributes
    env.state.chat_list = {"P0": [], "P1": []}
    env.state.behavior_list = {"P0": [], "P1": []}
    env.state.scene_list = {"P0": [], "P1": []}
    env.state.steps = {"0": 0, "1": 0}
    env.state.haschat = {"P0": False, "P1": False}
    env.state.haschat_other = {"P0": False, "P1": False}
    env.state.fail_back = {"P0": [], "P1": []}
    env.state.state_back = {"P0": [], "P1_done": [], "P1": []}
    
    return env

class ProAgentWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, seed, env_id, base_port, is_train, env_kwargs, layout_override=None):
        random.seed(seed + env_id)
        np.random.seed(seed + env_id)
        self.layout = layout_override if layout_override is not None else env_kwargs.get('layout', 'cramped_room')
        self.env = env_fn(seed, env_id, base_port, is_train, env_kwargs, layout_override=self.layout)
        self.env_id = env_id
        self.is_train = is_train
        self.debug = bool(env_kwargs.get("debug", False))  # 总开关，控制详细日志
        self.debug_print_rewards = bool(env_kwargs.get("debug_print_rewards", True))
        self.episode = 0
        self.timestep = 0
        self.total_reward = 0
        # Track cumulative rewards for Overcooked (for 'won' calculation)
        self.cumulative_sparse_reward = 0.0
        self.cumulative_shaped_reward = 0.0
    
    def step(self, actions, infos=None, step_info=None):
        """Execute a step in the environment"""
        if actions is None or len(actions) == 0:
            return step_info
        
        # 检查是否已完成（参考 cwah 的实现）
        # 如果 infos 中标记了 ml_action_completed，说明已完成
        # 已完成的环境直接返回 step_info，不执行实际的 env.step()，timestep 不变
        if infos is not None and step_info is not None:
            # infos 可能是 dict 格式 {0: info_p0, 1: info_p1}
            if isinstance(infos, dict) and infos.get(0) is not None:
                info_p0 = infos[0]
                if isinstance(info_p0, dict) and info_p0.get('ml_action_completed', False):
                    # 已完成，直接返回之前的 step_info，不执行 env.step()
                    # step_info 是元组 (obs, reward, done, info)
                    return step_info
        
        if infos is not None and isinstance(infos, dict):
            info_p1 = infos.get(1)
            if isinstance(info_p1, dict):
                state_updates = info_p1.get("state_updates")
                if isinstance(state_updates, dict):
                    for attr, value in state_updates.items():
                        setattr(self.env.state, attr, value)

        # actions should be a list of two actions [action_p0, action_p1]
        if isinstance(actions, dict):
            # Convert dict to list if needed
            joint_action = [actions.get(0), actions.get(1)]
        else:
            joint_action = actions
        
        # Ensure we have two actions
        if len(joint_action) < 2:
            joint_action = joint_action + [None] * (2 - len(joint_action))
        # 为了后续可能需要就地改写 joint_action（如碰撞规避），统一转成 list
        if not isinstance(joint_action, list):
            try:
                joint_action = list(joint_action)
            except Exception:
                joint_action = [joint_action]
        
        # 关键修复：Ray 里 driver 侧拿到的是 state 的序列化副本，agent.action(state) 修改不会回写到 worker。
        # 因此必须在 worker 侧用 projection/memory 传回来的 infos（action_infos）来同步当前的 ML Action，
        # 否则日志里的 Current ML Action 会一直是 N/A。
        if infos is not None and isinstance(infos, dict):
            try:
                # Ensure containers exist
                if not hasattr(self.env.state, "behavior_list") or self.env.state.behavior_list is None:
                    self.env.state.behavior_list = {"P0": [], "P1": []}
                if not hasattr(self.env.state, "ml_actions") or self.env.state.ml_actions is None:
                    self.env.state.ml_actions = [None, None]

                for player_idx in (0, 1):
                    info_i = infos.get(player_idx)
                    if not isinstance(info_i, dict):
                        continue

                    ml_action = info_i.get("ml_action")
                    if ml_action is not None:
                        key = f"P{player_idx}"
                        self.env.state.behavior_list.setdefault(key, [])
                        # 只在变化时追加，避免每个低阶 step 都重复刷同一个高阶动作
                        last = self.env.state.behavior_list[key][-1] if self.env.state.behavior_list[key] else None
                        last_plan = last.get("Plan") if isinstance(last, dict) else None
                        if last_plan != ml_action:
                            self.env.state.behavior_list[key].append({"Plan": ml_action})
                            # 控制长度，避免无限增长
                            if len(self.env.state.behavior_list[key]) > 200:
                                self.env.state.behavior_list[key] = self.env.state.behavior_list[key][-200:]

                    # 如果上层标记该 ML 动作已完成，把它写入 ml_actions 方便日志显示 “finished <...>”
                    if info_i.get("ml_action_completed", False) and info_i.get("ml_action") is not None:
                        self.env.state.ml_actions[player_idx] = info_i.get("ml_action")
            except Exception as e:
                if self.debug:
                    print(f"[Worker-{self.env_id}] Warning: failed to sync ml_action from infos: {e}", flush=True)

        # 关键修复：在 step 之前保存自定义属性（参考原始代码 main.py 第 119-126 行）
        # 执行流程（与原始代码 main.py 一致）：
        # 1. env_manager.step() 中调用 memory.get_actions()，传入 env.state
        # 2. get_actions() 中调用 agents[1].action(state)，ProAgent 会修改 state 的属性
        # 3. 这里保存修改后的属性（在 env.step() 之前）
        # 4. env.step() 会创建新的 state 对象
        # 5. 恢复属性到新的 state 对象
        current_state = self.env.state
        last_chat = getattr(current_state, 'chat_list', {"P0": [], "P1": []}).copy()
        last_behavior = getattr(current_state, 'behavior_list', {"P0": [], "P1": []}).copy()
        last_scenes = getattr(current_state, 'scene_list', {"P0": [], "P1": []}).copy()
        last_steps = getattr(current_state, 'steps', {"0": 0, "1": 0}).copy()
        last_haschat = getattr(current_state, 'haschat', {"P0": False, "P1": False}).copy()
        last_haschat_other = getattr(current_state, 'haschat_other', {"P0": False, "P1": False}).copy()
        last_fail = getattr(current_state, 'fail_back', {"P0": [], "P1": []}).copy()
        last_state_back = getattr(current_state, 'state_back', {"P0": [], "P1_done": [], "P1": []}).copy()
        last_ml_actions = getattr(current_state, 'ml_actions', [None, None])
        
        # ======== 碰撞规避（关键修复）========
        # 现象：agent0/agent1 规划到相同“下一步目标格”，Overcooked 的碰撞处理会让两人都回退，
        # 导致位置永远不变但 timestep 一直增长（看起来“卡死”）。
        # 策略：如果两人下一步意图位置相同，则让 agent1 让路（STAY），保证 agent0 优先前进。
        try:
            if isinstance(joint_action, list) and len(joint_action) >= 2:
                a0, a1 = joint_action[0], joint_action[1]
                mdp = getattr(self.env, "mdp", None)
                if mdp is not None and hasattr(current_state, "players") and len(current_state.players) >= 2:
                    p0_pos = current_state.players[0].position
                    p1_pos = current_state.players[1].position
                    try:
                        valid_positions = mdp.get_valid_player_positions()
                    except Exception:
                        valid_positions = None

                    def _intended_next_pos(pos, a):
                        if a is None:
                            return pos
                        if a == Action.INTERACT or a == Action.STAY:
                            return pos
                        try:
                            new_pos = Action.move_in_direction(pos, a)
                        except Exception:
                            return pos
                        if valid_positions is not None and new_pos not in valid_positions:
                            return pos
                        return new_pos

                    p0_next = _intended_next_pos(p0_pos, a0)
                    p1_next = _intended_next_pos(p1_pos, a1)

                    # 两人下一步目标格相同：让 agent1 STAY（仅对 agent1 生效）
                    if p0_next == p1_next and a1 != Action.STAY:
                        joint_action[1] = Action.STAY
                        # 将“让路”信息写入 infos，便于日志/排查
                        if infos is not None and isinstance(infos, dict):
                            infos.setdefault(1, {})
                            if isinstance(infos.get(1), dict):
                                infos[1]["yield_to_p0"] = True
                                infos[1]["original_action"] = a1
        except Exception as e:
            # 绝不因为规避逻辑影响正常 step
            if self.debug:
                print(f"[Worker-{self.env_id}] Warning: collision-avoidance failed: {e}", flush=True)
        
        # 执行 step（会创建新的 state 对象）
        # 检查环境是否已结束，避免对已 done 的环境调用 step
        if self.env.is_done():
            obs = self.env.state
            reward = 0
            done = True
            env_info = {"episode": {"ep_sparse_r": self.cumulative_sparse_reward, "ep_shaped_r": self.cumulative_shaped_reward}}
        else:
            obs, reward, done, env_info = self.env.step(joint_action)
        
        # 关键修复：将保存的属性重新赋值给新的 state（参考原始代码 main.py 第 129-135 行）
        # 注意：obs 就是 self.env.state，它们是同一个对象
        new_state = self.env.state
        new_state.chat_list = last_chat
        new_state.behavior_list = last_behavior
        new_state.scene_list = last_scenes
        new_state.steps = last_steps
        new_state.haschat = last_haschat
        new_state.haschat_other = last_haschat_other
        new_state.fail_back = last_fail
        new_state.state_back = last_state_back
        new_state.ml_actions = last_ml_actions
        
        # 关键修复：递增步数计数器（参考原始代码 main.py 第 137-138 行）
        new_state.steps["0"] += 1
        new_state.steps["1"] += 1
        
        # 更新总奖励和累计奖励
        # OvercookedEnv 返回的 reward 是 sparse_reward（只有交付汤时才有）
        # 我们需要跟踪累计奖励以便在 episode 结束时计算总得分
        self.total_reward += reward
        # 从 env_info 中获取 shaped_r（如果可用），否则使用 0
        shaped_r = env_info.get('shaped_r', 0.0) if isinstance(env_info, dict) else 0.0
        self.cumulative_sparse_reward += reward  # reward 是 sparse_reward
        self.cumulative_shaped_reward += shaped_r
        
        # 递增 timestep（在记录日志之前，这样日志显示的是当前步骤）
        if not done:
            self.timestep += 1
        
        # 添加详细日志输出（参考原始日志格式）
        # 注意：在 Ray 环境中，print 输出会带有前缀，确保日志可见
        if self.debug:
            self._log_step_info(current_state, new_state, joint_action, reward, shaped_r, done, env_info)
        
        # 如果 episode 结束，重置计数器
        if done:
            # 确保 env_info 包含 episode 信息（用于计算 'won'）
            if isinstance(env_info, dict):
                # 如果 OvercookedEnv 没有提供 episode 信息，我们自己添加
                if 'episode' not in env_info:
                    env_info['episode'] = {
                        'ep_sparse_r': self.cumulative_sparse_reward,
                        'ep_shaped_r': self.cumulative_shaped_reward,
                        'ep_length': self.timestep + 1  # +1 because timestep starts at 0
                    }
            self.episode += 1
            self.timestep = 0
            self.total_reward = 0
            # 重置累计奖励
            self.cumulative_sparse_reward = 0.0
            self.cumulative_shaped_reward = 0.0
        
        # 确保返回的 obs 也是更新后的 state（obs 和 self.env.state 是同一个对象）
        if self.debug and self.debug_print_rewards:
            try:
                # Print a compact reward line right before returning
                ep = env_info.get("episode") if isinstance(env_info, dict) else None
                ep_sparse = ep.get("ep_sparse_r") if isinstance(ep, dict) else None
                ep_shaped = ep.get("ep_shaped_r") if isinstance(ep, dict) else None
                print(
                    f"[Worker-{self.env_id}] return reward: sparse={float(reward):.3f} shaped={float(shaped_r):.3f} "
                    f"cum_sparse={float(self.cumulative_sparse_reward):.3f} cum_shaped={float(self.cumulative_shaped_reward):.3f} "
                    f"done={done} ep_sparse={ep_sparse} ep_shaped={ep_shaped}",
                    flush=True,
                )
            except Exception:
                pass
        return new_state, reward, done, env_info
    
    def _log_step_info(self, prev_state, current_state, joint_action, reward, shaped_r, done, env_info):
        """输出详细的步骤信息日志"""
        try:
            # 获取玩家信息
            players = current_state.players
            horizon = self.env.horizon
            
            # 获取当前 ML Action（从 behavior_list 中获取最后一个 Plan）
            behavior_list = getattr(current_state, 'behavior_list', {"P0": [], "P1": []})
            ml_action_p0 = "N/A"
            ml_action_p1 = "N/A"
            
            if behavior_list.get("P0") and len(behavior_list["P0"]) > 0:
                last_plan = behavior_list["P0"][-1]
                if isinstance(last_plan, dict) and "Plan" in last_plan:
                    ml_action_p0 = last_plan["Plan"]
            
            if behavior_list.get("P1") and len(behavior_list["P1"]) > 0:
                last_plan = behavior_list["P1"][-1]
                if isinstance(last_plan, dict) and "Plan" in last_plan:
                    ml_action_p1 = last_plan["Plan"]
            
            # 获取玩家位置、方向、持有物品
            def get_player_info(player, player_idx):
                pos = player.position
                orientation = player.orientation
                held_obj = player.held_object
                if held_obj is None:
                    holding = "nothing"
                else:
                    holding = held_obj.name if hasattr(held_obj, 'name') else str(held_obj)
                
                # 方向转字符串
                ori_map = {
                    Direction.NORTH: "(0, -1)",
                    Direction.SOUTH: "(0, 1)",
                    Direction.EAST: "(1, 0)",
                    Direction.WEST: "(-1, 0)"
                }
                ori_str = ori_map.get(orientation, str(orientation))
                
                return pos, ori_str, holding
            
            pos_p0, ori_p0, holding_p0 = get_player_info(players[0], 0)
            pos_p1, ori_p1, holding_p1 = get_player_info(players[1], 1)
            
            # 获取执行的动作
            def action_to_string(action, player_idx):
                if action is None:
                    return "N/A"
                try:
                    char = Action.to_char(action)
                    # 如果是 interact，尝试从 ml_actions 中获取详细信息
                    if action == Action.INTERACT:
                        # 从当前 state 的 ml_actions 中获取（step 后更新的）
                        ml_actions = getattr(current_state, 'ml_actions', [None, None])
                        if ml_actions[player_idx] is not None:
                            return f"interact ({ml_actions[player_idx]})"
                        return "interact"
                    return char
                except:
                    return str(action)
            
            action_p0_str = action_to_string(joint_action[0], 0)
            action_p1_str = action_to_string(joint_action[1], 1)
            
            # 获取完成的 ML Actions
            ml_actions = getattr(current_state, 'ml_actions', [None, None])
            completed_ml_actions = []
            for i, ml_action in enumerate(ml_actions):
                if ml_action is not None:
                    completed_ml_actions.append(f"P{i} finished <{ml_action}>.")
            completed_ml_str = " ".join(completed_ml_actions) if completed_ml_actions else ""
            
            # 获取地图字符串（使用新的状态）
            try:
                map_string = self.env.mdp.state_string(self.env.state)
            except Exception as e:
                map_string = f"[Error rendering map: {e}]"
            
            # 输出日志（使用 sys.stdout 确保在 Ray 环境中正确输出）
            log_lines = [
                f"\n{'='*80}",
                f"[Worker-{self.env_id}] Episode {self.episode + 1}/1 | Timestep {self.timestep}/{horizon}",
                f"{'='*80}",
                f"\n【地图 (Map)】",
                f"{'-'*80}",
                map_string,
                f"{'-'*80}",
                f"\nPlayer 0 (ProAgent):",
                f"  Position: {pos_p0} | Orientation: {ori_p0}",
                f"  Holding: {holding_p0}",
                f"  Current ML Action: {ml_action_p0}",
                f"Player 1 (ProAgent):",
                f"  Position: {pos_p1} | Orientation: {ori_p1}",
                f"  Holding: {holding_p1}",
                f"  Current ML Action: {ml_action_p1}",
                f"\nActions Executed:",
                f"  P0: {action_p0_str}",
                f"  P1: {action_p1_str}",
            ]
            # 总是显示 Completed ML Actions 行，即使为空
            if completed_ml_str:
                log_lines.append(f"Completed ML Actions: {completed_ml_str} ")
            log_lines.append(
                f"Reward(sparse): {reward} | Shaped: {shaped_r} | Total Sparse (episode): {self.total_reward} "
                f"| Cum Shaped (episode): {self.cumulative_shaped_reward}"
            )

            # If done, print episode totals if available
            if done and isinstance(env_info, dict) and "episode" in env_info:
                ep = env_info["episode"]
                if isinstance(ep, dict):
                    log_lines.append(
                        f"Episode totals: ep_sparse_r={ep.get('ep_sparse_r', 0.0)} ep_shaped_r={ep.get('ep_shaped_r', 0.0)} "
                        f"ep_length={ep.get('ep_length', 'N/A')}"
                    )
            
            # 输出所有日志行并刷新（确保完整输出）
            output_str = "\n".join(log_lines)
            print(output_str, flush=True)
        except Exception as e:
            # 如果日志输出出错，打印错误信息以便调试
            print(f"Error in _log_step_info: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
    
    def reset(self):
        """Reset the environment"""
        obs = self.env.reset()
        
        # 重置计数器
        self.timestep = 0
        self.total_reward = 0
        # Reset cumulative rewards
        self.cumulative_sparse_reward = 0.0
        self.cumulative_shaped_reward = 0.0
        
        # Reset state attributes
        self.env.state.chat_list = {"P0": [], "P1": []}
        self.env.state.behavior_list = {"P0": [], "P1": []}
        self.env.state.scene_list = {"P0": [], "P1": []}
        self.env.state.steps = {"0": 0, "1": 0}
        self.env.state.haschat = {"P0": False, "P1": False}
        self.env.state.haschat_other = {"P0": False, "P1": False}
        self.env.state.fail_back = {"P0": [], "P1": []}
        self.env.state.state_back = {"P0": [], "P1_done": [], "P1": []}
        
        infos = {
            'layout': self.layout,  # Use the layout name we used to create this env (not mdp.layout_name which may differ)
            'mdp': self.env.mdp,
            'horizon': self.env.horizon,
        }
        return obs, infos
    
    def get_observations(self):
        """Get current observations"""
        return self.env.state
    
    def get_state(self):
        """Get current state"""
        return self.env.state


class ProAgentEnvs(gym.Env):
    def __init__(self, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        self.env_kwargs = env_kwargs

        random.seed(seed)
        np.random.seed(seed)

        # Get layouts configuration
        # If env_kwargs['layouts'] is provided, use it; otherwise use ALL_LAYOUTS
        # If env_kwargs['layout'] is a single string, use it for all envs (backward compatible)
        layouts_config = env_kwargs.get('layouts', None)
        single_layout = env_kwargs.get('layout', None)
        
        if layouts_config is not None:
            # Multi-layout mode: layouts is a list
            if isinstance(layouts_config, str):
                layouts_config = [layouts_config]
            self.layouts = list(layouts_config)
        elif single_layout is not None:
            # Single layout mode (backward compatible)
            self.layouts = [single_layout]
        else:
            # Default: use all layouts
            self.layouts = ALL_LAYOUTS.copy()
        
        print(f"[ProAgentEnvs] Initializing with {env_num} envs x {group_n} groups = {self.num_processes} total workers")
        print(f"[ProAgentEnvs] Available layouts: {self.layouts}")

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(ProAgentWorker)
        self.workers = []
        self.worker_layouts = []  # Track which layout each worker uses
        
        for i in range(self.num_processes):
            # Determine layout for this worker based on env_id and group_n
            layout_for_env = get_layout_for_env(i, group_n, self.layouts)
            self.worker_layouts.append(layout_for_env)
            
            worker = env_worker.remote(
                seed, i, np.random.randint(11000, 13000), is_train, env_kwargs,
                layout_override=layout_for_env
            )
            self.workers.append(worker)
        
        # Print layout distribution
        layout_counts = {}
        for layout in self.worker_layouts:
            layout_counts[layout] = layout_counts.get(layout, 0) + 1
        print(f"[ProAgentEnvs] Layout distribution: {layout_counts}")

    def step(self, actions, infos=None, step_info=None):
        """
        Execute a step in all environments.
        actions: list of actions for each environment, each action is [action_p0, action_p1]
        infos: list of action infos for each environment
        step_info: previous step info tuple (obs_list, rewards_list, dones_list, infos_list)
        """
        assert len(actions) == self.num_processes, \
            f"The num of actions ({len(actions)}) must be equal to the num of processes ({self.num_processes})"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            # 检查是否已完成（参考 cwah 的实现）
            # 如果 action[0] 是 STAY 且 infos 中标记了 ml_action_completed，传递 step_info
            env_infos = infos[i] if infos is not None and i < len(infos) else None
            if env_infos is not None and env_infos.get(0) is not None:
                info_p0 = env_infos[0]
                if info_p0.get('ml_action_completed', False) and step_info is not None:
                    # 已完成的环境，传递之前的 step_info，让 worker 直接返回
                    # 这样不会执行实际的 env.step()，timestep 不变
                    future = worker.step.remote(actions[i], env_infos, (
                        step_info[0][i] if i < len(step_info[0]) else None,
                        step_info[1][i] if i < len(step_info[1]) else 0.0,
                        step_info[2][i] if i < len(step_info[2]) else False,
                        step_info[3][i] if i < len(step_info[3]) else {}
                    ))
                else:
                    # 未完成的环境，正常执行
                    future = worker.step.remote(actions[i], env_infos)
            else:
                # 没有 infos，正常执行
                future = worker.step.remote(actions[i], env_infos)

            futures.append(future)

        # Collect results
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        results = ray.get(futures)

        for i, (obs, reward, done, info) in enumerate(results):
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            infos_list.append(info)
        
        return obs_list, rewards_list, dones_list, infos_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        obs_list = []
        infos_list = []

        # Send reset to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            infos_list.append(info)

        return obs_list, infos_list
    
    def get_observations(self):
        """Get observations from all environments"""
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.get_observations.remote()
            futures.append(future)

        obs_list = []
        results = ray.get(futures)
        for i, obs in enumerate(results):
            obs_list.append(obs)
        return obs_list
    
    def get_state(self):
        """Get states from all environments"""
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.get_state.remote()
            futures.append(future)
        
        state_list = []
        results = ray.get(futures)
        for i, state in enumerate(results):
            state_list.append(state)
        return state_list

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_proagent_envs(seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return ProAgentEnvs(seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)

