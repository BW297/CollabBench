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

from typing import List
import sys
import os

# Add lib to path for overcooked_ai import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from overcooked_ai_py.mdp.actions import Action

def proagent_projection(actions: List[str], memory, obs, goal, action_infos, cnt, available_actions, debug=False):
    """
    A function to process the actions for ProAgent environment.
    
    Args:
        actions: List of text actions from RL agent (for p0)
        memory: proagent_agent_memory instance
        obs: Current observations (states)
        goal: Not used for Overcooked
        action_infos: Previous action infos
        cnt: Step counter
        available_actions: Available actions (not used for Overcooked)
        debug: Whether to print debug info
    
    Returns:
        actions: List of joint actions [action_p0, action_p1] where each is an Action object
        valids: List of validity flags
        infos: List of action infos
    """
    # ========== 调试信息：函数输入参数 ==========
    if debug:
        print("\n" + "="*80)
        print("🔍 [proagent_projection] 开始处理动作")
        print("="*80)
        print(f"输入参数:")
        print(f"  actions (RL文本动作): {actions}")
        print(f"  obs 类型: {type(obs)}, 数量: {len(obs) if isinstance(obs, list) else 'N/A'}")
        print(f"  goal: {goal}")
        print(f"  action_infos: {action_infos}")
        print(f"  cnt (步数): {cnt}")
        print(f"  available_actions: {available_actions}")
        print("-"*80)
    
    # 直接调用 memory（内部 Ray actor 完成 p0/p1 的动作生成）
    actions, infos = memory.get_actions(actions, obs, goal, action_infos, cnt)
    
    # ========== 调试信息：memory.get_actions 返回结果 ==========
    if debug:
        print(f"\n🔍 [proagent_projection] memory.get_actions 返回结果:")
        print(f"  actions 类型: {type(actions)}, 数量: {len(actions) if isinstance(actions, list) else 'N/A'}")
        print(f"  infos 类型: {type(infos)}, 数量: {len(infos) if isinstance(infos, list) else 'N/A'}")
        if isinstance(actions, list) and len(actions) > 0:
            print(f"  第一个 action_dict: {actions[0]}, 类型: {type(actions[0])}")
        print("-"*80)
    
    # Convert actions to proper format
    # actions from memory should be a list of dicts: [{0: action_p0, 1: action_p1}, ...]
    # We need to convert to list of lists: [[action_p0, action_p1], ...]
    joint_actions = []
    valids = []
    
    if debug:
        print(f"\n🔍 [proagent_projection] 开始转换动作格式")
        print(f"  处理 {len(actions)} 个环境的动作")
        print("-"*80)
    
    for i, action_dict in enumerate(actions):
        if debug:
            print(f"\n[环境 {i}] 处理动作字典:")
            print(f"  action_dict: {action_dict}, 类型: {type(action_dict)}")
        
        if isinstance(action_dict, dict):
            # Extract actions for both players
            action_p0 = action_dict.get(0)
            action_p1 = action_dict.get(1)
            
            if debug:
                print(f"  提取的原始动作:")
                print(f"    action_p0: {action_p0}, 类型: {type(action_p0)}")
                print(f"    action_p1: {action_p1}, 类型: {type(action_p1)}")
            # Ensure actions are valid (can be Direction tuple, Action.STAY tuple, or Action.INTERACT string)
            # Check if action is already a valid action (in Action.ALL_ACTIONS)
            if debug:
                print(f"  验证和转换 action_p0:")
            if action_p0 is None:
                if debug:
                    print(f"    ⚠️  action_p0 为 None")
            elif action_p0 in Action.ALL_ACTIONS:
                if debug:
                    print(f"    ✅ action_p0 已经是有效动作: {action_p0}")
            else:
                if debug:
                    print(f"    ⚠️  action_p0 不在 ALL_ACTIONS 中，需要转换")
                # Try to convert if it's a string or int
                if isinstance(action_p0, str):
                    if debug:
                        print(f"    尝试将字符串 '{action_p0}' 转换为动作")
                    # Try to map string to action using INDEX_TO_ACTION
                    try:
                        # Check if it's an action name
                        if action_p0.upper() in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
                            from overcooked_ai_py.mdp.actions import Direction
                            action_p0 = getattr(Direction, action_p0.upper())
                            if debug:
                                print(f"    ✅ 转换为方向: {action_p0}")
                        elif action_p0.upper() in ['STAY', 'INTERACT']:
                            action_p0 = getattr(Action, action_p0.upper())
                            if debug:
                                print(f"    ✅ 转换为动作: {action_p0}")
                        else:
                            action_p0 = Action.STAY
                            if debug:
                                print(f"    ⚠️  无法识别，使用默认: Action.STAY")
                    except Exception as e:
                        action_p0 = Action.STAY
                        if debug:
                            print(f"    ❌ 转换失败: {e}, 使用默认: Action.STAY")
                elif isinstance(action_p0, int):
                    if debug:
                        print(f"    尝试将整数 {action_p0} 转换为动作")
                    if 0 <= action_p0 < len(Action.INDEX_TO_ACTION):
                        original_action_p0 = action_p0
                        action_p0 = Action.INDEX_TO_ACTION[action_p0]
                        if debug:
                            print(f"    ✅ 通过索引转换: INDEX_TO_ACTION[{original_action_p0}] = {action_p0}")
                    else:
                        action_p0 = Action.STAY
                        if debug:
                            print(f"    ⚠️  索引 {action_p0} 超出范围，使用默认: Action.STAY")
                else:
                    action_p0 = Action.STAY
                    if debug:
                        print(f"    ⚠️  未知类型 {type(action_p0)}，使用默认: Action.STAY")
            
            if debug:
                print(f"  验证和转换 action_p1:")
            if action_p1 is None:
                if debug:
                    print(f"    ⚠️  action_p1 为 None")
            elif action_p1 in Action.ALL_ACTIONS:
                if debug:
                    print(f"    ✅ action_p1 已经是有效动作: {action_p1}")
            else:
                if debug:
                    print(f"    ⚠️  action_p1 不在 ALL_ACTIONS 中，需要转换")
                if isinstance(action_p1, str):
                    if debug:
                        print(f"    尝试将字符串 '{action_p1}' 转换为动作")
                    try:
                        if action_p1.upper() in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
                            from overcooked_ai_py.mdp.actions import Direction
                            action_p1 = getattr(Direction, action_p1.upper())
                            if debug:
                                print(f"    ✅ 转换为方向: {action_p1}")
                        elif action_p1.upper() in ['STAY', 'INTERACT']:
                            action_p1 = getattr(Action, action_p1.upper())
                            if debug:
                                print(f"    ✅ 转换为动作: {action_p1}")
                        else:
                            action_p1 = Action.STAY
                            if debug:
                                print(f"    ⚠️  无法识别，使用默认: Action.STAY")
                    except Exception as e:
                        action_p1 = Action.STAY
                        if debug:
                            print(f"    ❌ 转换失败: {e}, 使用默认: Action.STAY")
                elif isinstance(action_p1, int):
                    if debug:
                        print(f"    尝试将整数 {action_p1} 转换为动作")
                    if 0 <= action_p1 < len(Action.INDEX_TO_ACTION):
                        original_action_p1 = action_p1
                        action_p1 = Action.INDEX_TO_ACTION[action_p1]
                        if debug:
                            print(f"    ✅ 通过索引转换: INDEX_TO_ACTION[{original_action_p1}] = {action_p1}")
                    else:
                        action_p1 = Action.STAY
                        if debug:
                            print(f"    ⚠️  索引 {action_p1} 超出范围，使用默认: Action.STAY")
                else:
                    action_p1 = Action.STAY
                    if debug:
                        print(f"    ⚠️  未知类型 {type(action_p1)}，使用默认: Action.STAY")
            
            if debug:
                print(f"  最终动作:")
                print(f"    action_p0: {action_p0}, 类型: {type(action_p0)}, 在 ALL_ACTIONS 中: {action_p0 in Action.ALL_ACTIONS if action_p0 is not None else False}")
                print(f"    action_p1: {action_p1}, 类型: {type(action_p1)}, 在 ALL_ACTIONS 中: {action_p1 in Action.ALL_ACTIONS if action_p1 is not None else False}")
            
            joint_actions.append([action_p0, action_p1])
            # ------------------------------------------------------------
            # VALID definition (agent0 only):
            # - Only validate agent0 (p0) action / intent for training.
            # - "wait" is ALWAYS invalid (per user requirement).
            # - If agent0 set _force_ml_action_done=True (early termination), it's invalid.
            # ------------------------------------------------------------
            valid = 1
            try:
                # 1) low-level legality for agent0
                if action_p0 is None or action_p0 not in Action.ALL_ACTIONS:
                    valid = 0

                # 2) high-level "wait" is invalid (look at ml_action in infos)
                info_env = infos[i] if isinstance(infos, list) and i < len(infos) else None
                info_p0 = info_env.get(0) if isinstance(info_env, dict) else None
                ml_action_p0 = None
                if isinstance(info_p0, dict):
                    ml_action_p0 = info_p0.get("ml_action") or info_p0.get("plan") or info_p0.get("Plan")
                if ml_action_p0 is not None and "wait" in str(ml_action_p0).lower():
                    valid = 0
                    if isinstance(info_p0, dict):
                        info_p0.setdefault("invalid_reason", "wait")

                # 3) force-terminate flag from ProAgent (agent0) is invalid
                agent0 = None
                if hasattr(memory, "_data") and memory._data is not None and i < len(memory._data):
                    try:
                        agent0 = memory._data[i].get(0) if isinstance(memory._data[i], dict) else None
                    except Exception:
                        agent0 = None
                if agent0 is not None and getattr(agent0, "_force_ml_action_done", False):
                    valid = 0
                    if isinstance(info_p0, dict):
                        info_p0.setdefault("invalid_reason", "force_ml_action_done")
            except Exception as e:
                # Fail safe: don't block execution
                valid = 1

            valids.append(valid)
            if debug:
                print(f"  添加到 joint_actions: {[action_p0, action_p1]}, valid(p0_only): {valid}")
        else:
            # Fallback: assume single action for p0, p1 will be handled by memory
            if action_dict in Action.ALL_ACTIONS:
                joint_actions.append([action_dict, None])
            elif isinstance(action_dict, str):
                # Try to map string to action
                try:
                    if action_dict.upper() in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
                        from overcooked_ai_py.mdp.actions import Direction
                        action = getattr(Direction, action_dict.upper())
                    elif action_dict.upper() in ['STAY', 'INTERACT']:
                        action = getattr(Action, action_dict.upper())
                    else:
                        action = Action.STAY
                except:
                    action = Action.STAY
                joint_actions.append([action, None])
            elif isinstance(action_dict, int):
                if 0 <= action_dict < len(Action.INDEX_TO_ACTION):
                    action = Action.INDEX_TO_ACTION[action_dict]
                else:
                    action = Action.STAY
                joint_actions.append([action, None])
            else:
                joint_actions.append([Action.STAY, None])
            # In fallback mode, still apply p0-only validity rules best-effort
            valid = 1
            try:
                action_p0 = joint_actions[-1][0]
                if action_p0 is None or action_p0 not in Action.ALL_ACTIONS:
                    valid = 0

                info_env = infos[i] if isinstance(infos, list) and i < len(infos) else None
                info_p0 = info_env.get(0) if isinstance(info_env, dict) else None
                ml_action_p0 = None
                if isinstance(info_p0, dict):
                    ml_action_p0 = info_p0.get("ml_action") or info_p0.get("plan") or info_p0.get("Plan")
                if ml_action_p0 is not None and "wait" in str(ml_action_p0).lower():
                    valid = 0
                    if isinstance(info_p0, dict):
                        info_p0.setdefault("invalid_reason", "wait")

                agent0 = None
                if hasattr(memory, "_data") and memory._data is not None and i < len(memory._data):
                    try:
                        agent0 = memory._data[i].get(0) if isinstance(memory._data[i], dict) else None
                    except Exception:
                        agent0 = None
                if agent0 is not None and getattr(agent0, "_force_ml_action_done", False):
                    valid = 0
                    if isinstance(info_p0, dict):
                        info_p0.setdefault("invalid_reason", "force_ml_action_done")
            except Exception:
                valid = 1
            valids.append(valid)
            if debug:
                print(f"  [环境 {i}] Fallback 处理完成")
    
    # ========== 调试信息：最终返回结果 ==========
    if debug:
        print("\n" + "-"*80)
        print("🔍 [proagent_projection] 最终返回结果:")
        print(f"  joint_actions 数量: {len(joint_actions)}")
        print(f"  valids: {valids}")
        if len(joint_actions) > 0:
            print(f"  第一个 joint_action: {joint_actions[0]}")
            if len(joint_actions[0]) == 2:
                print(f"    action_p0: {joint_actions[0][0]}, 类型: {type(joint_actions[0][0])}")
                print(f"    action_p1: {joint_actions[0][1]}, 类型: {type(joint_actions[0][1])}")
        print("="*80 + "\n")
    
    return joint_actions, valids, infos

