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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import pickle
import numpy as np
from functools import partial
import os
from types import SimpleNamespace
from verl.utils.interactivity import InteractivityMetric
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory, cwah_agent_memory, proagent_agent_memory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs
    
class CwahEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = cwah_agent_memory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(obs), obs=obs, infos=infos, base_url=self.config.env.cwah.base_url, lm_id=self.config.env.cwah.lm_id, prompt_template = CWAH_ACTION_TEMPLATE, sampling_parameters=self.config.env.cwah.sampling_parameters)
        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': self.build_anchor_obs()
        }
        self.steps = [0] * len(obs)
        return observations, infos

    def step(self, text_actions: List[str]):
        all_done = False
        
        action_infos = []
        rewards = [0.0] * self.envs.num_processes
        step_info = None
        cnt = 0
        available_actions = self.build_available_actions()
        while all_done == False:
            obs = self.envs.get_observations()
            goal = self.envs.get_goal() 
            actions, valids, action_infos = self.projection_f(text_actions, self.memory, obs, goal, action_infos, cnt, available_actions)
            step_info = self.envs.step(actions, action_infos, step_info)
            # print(" Cnt: {}, Step Rewards: {}, Sum Rewards: {}".format(cnt, step_info[1], rewards))
            # communication reward
            if cnt == 0:
                for i in range(len(rewards)):
                    if (not (len(actions[i]) == 0 or (actions[i][0] is None and action_infos[i][0]['plan'] != "[wait]"))) and valids[i] and step_info[2][i] == False and actions[i][0] is not None and '[send_message]' in actions[i][0]:
                        rewards[i] = rewards[i] + 0.05
            # normal reward
            rewards = [rewards[i] if (len(actions[i]) == 0 or (actions[i][0] is None and action_infos[i][0]['plan'] != "[wait]")) else rewards[i] + step_info[1][i] for i in range(len(rewards))]
            # print(" Cnt: {}, Step Rewards: {}, Sum Rewards: {}".format(cnt, step_info[1], rewards))
            all_done = all(((action_infos[i][0]['plan'] is None) or step_info[2][i]) for i in range(len(action_infos)))
            for i in range(self.envs.num_processes):
                if step_info[2][i] and self.steps[i] == 0:
                    self.steps[i] = self.envs.get_step()[i]
            cnt += 1
        
        next_obs, _, dones, infos, messages = step_info
        # print("Dones:", dones)
        # for i, info in enumerate(infos):
        #     if step_info[2][i] and self.steps[i] == 0:
        #         self.steps[i] = self.envs.get_step()[i]    

        # print(self.steps)
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            if self.steps[i] >= 250:
                info['won'] = 250
            else:
                info['won'] = self.steps[i]
            info['step'] = self.steps[i]

        self.pre_text_obs = next_obs
        next_observations = {
            'text': self.build_text_obs(),
            'image': None, 
            'anchor': self.build_anchor_obs()
        }
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        obs = self.envs.get_observations()
        goal = self.envs.get_goal() 
        total_obs = self.memory.build_text_obs(obs, goal)
        text_obs = []
        for item in total_obs:
            text_obs.append(item[0])
        return text_obs


    def build_available_actions(self) -> List[str]:
        """
        This function builds the available actions for the agent.
        """
        obs = self.envs.get_observations()
        goal = self.envs.get_goal() 
        total_obs = self.memory.build_text_obs(obs, goal)
        available_actions = []
        for item in total_obs:
            available_actions.append(item[1])
        return available_actions
    
    def build_anchor_obs(self) -> List[str]:
        """
        This function builds the anchor observation for the agent.
        """
        obs = self.envs.get_observations()
        goal = self.envs.get_goal() 

        return self.memory.build_anchor_obs(obs, goal)
        # obs_list = self.envs.get_observations()
        # anchor_obs_list = []
        # for obs in obs_list:
        #     anchor_obs = {}
        #     for agent_id, agent_obs in obs.items():
        #         clean_agent_obs = {
        #             k: v for k, v in agent_obs.items() if k != "messages" and k != "location"
        #         }
        #         anchor_obs[agent_id] = clean_agent_obs
        #     anchor_obs_list.append(anchor_obs)

        #         return anchor_obs_list

class ProAgentEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = proagent_agent_memory()
        super().__init__(envs, projection_f, config)
        # Get p0 and p1 from config
        self.p0 = getattr(config.env.proagent, 'p0', 'RL')
        self.p1 = getattr(config.env.proagent, 'p1', 'ProAgent')
        self.debug = getattr(config.env.proagent, 'debug', False)
        self._interactivity_metric = None

        # Reward shaping config (dense reward = sparse + shaped + extras).
        # Defaults are intentionally conservative; override via config.env.proagent.reward.*
        self._reward_cfg = self._get_proagent_reward_cfg()

    def _get_proagent_reward_cfg(self) -> Dict[str, Any]:
        """Read reward config from OmegaConf (if present) with safe defaults."""
        defaults: Dict[str, Any] = {
            # Mix in env_info['shaped_r'] (from OvercookedEnv) into the returned reward.
            "include_shaped_r": True,
            "shaped_coef": 1.0,

            # Extra shaping / penalties (applied per low-level env.step executed).
            "step_penalty": -0,          # small time penalty, ~ -4 over 400 steps
            "stay_penalty": -0.5,         # discourage idle, but keep small (STAY is sometimes optimal)
            "invalid_action_penalty": 0,  # disabled: actions are now filtered to be valid, no penalty needed

            # Set message reward 
            "set_message_reward": 0.01,    # reward for sending messages to teammate

            # Global scale / clip
            "reward_scale": 0.01,
            "reward_clip": [-0.5,1],            # e.g. [-10.0, 10.0]

            # Debug
            "debug_reward_breakdown": False,
            "debug_print_each_low_step": False,   # print per internal low-level env.step
            "debug_print_return": True,          # print right before returning from manager.step
            "debug_print_max_envs": 3,            # limit how many envs to print per line
            "debug_print_every_n": 1,             # print frequency over internal cnt
        }

        cfg = None
        try:
            cfg = getattr(self.config.env.proagent, "reward", None)
        except Exception:
            cfg = None

        if cfg is None:
            return defaults

        try:
            from omegaconf import OmegaConf
            cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
        except Exception:
            cfg_dict = cfg if isinstance(cfg, dict) else {}

        if not isinstance(cfg_dict, dict):
            return defaults

        merged = defaults.copy()
        merged.update({k: v for k, v in cfg_dict.items() if v is not None})
        return merged

    class _DummyTokenizer:
        def decode(self, responses, skip_special_tokens=True):
            if isinstance(responses, str):
                return responses
            try:
                return " ".join(str(x) for x in responses)
            except Exception:
                return str(responses)

    def _get_interactivity_metric(self):
        if self._interactivity_metric is not None:
            return self._interactivity_metric

        base_url = getattr(self.config.env.proagent, 'base_url', None)
        lm_id = getattr(self.config.env.proagent, 'lm_id', None)
        api_key = (
            os.environ.get("PROAGENT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )

        sampling_parameters_raw = getattr(self.config.env.proagent, 'sampling_parameters', {}) or {}
        try:
            from omegaconf import OmegaConf
            if hasattr(sampling_parameters_raw, '_content'):
                sampling_parameters = OmegaConf.to_container(sampling_parameters_raw, resolve=True)
            else:
                sampling_parameters = sampling_parameters_raw
        except Exception:
            sampling_parameters = sampling_parameters_raw

        sampling_parameters = sampling_parameters if isinstance(sampling_parameters, dict) else {}
        sampling_parameters = {
            "max_tokens": sampling_parameters.get("max_tokens", 256),
            "t": sampling_parameters.get("t", 0.0),
            "top_p": sampling_parameters.get("top_p", 1.0),
            "n": sampling_parameters.get("n", 1),
        }

        try:
            self._interactivity_metric = InteractivityMetric(
                num_retries=1,
                retry_after=1,
                tokenizer=self._DummyTokenizer(),
                lm_id=lm_id,
                api_key=api_key,
                base_url=base_url,
                sampling_parameters=sampling_parameters,
            )
        except Exception:
            self._interactivity_metric = None

        return self._interactivity_metric

    def _compute_interactivity_score(self, metric, observation_text: str, action_text: str, env_idx: int = None) -> float:
        if metric is None:
            return 0.0
        try:
            # 优先使用 agent0.profile（通过 memory 获取）
            profile = None
            if env_idx is not None and hasattr(self.memory, "get_p0_profile"):
                try:
                    profile = self.memory.get_p0_profile(env_idx)
                except Exception:
                    pass
            
            # Fallback: 使用 persona_name
            if not profile and env_idx is not None and hasattr(self.memory, "get_persona_name"):
                try:
                    profile = self.memory.get_persona_name(env_idx)
                except Exception:
                    pass
            
            # Fallback: 使用配置
            if not profile:
                profile = getattr(self.config.env.proagent, "profile_id", "default")

            data_item = SimpleNamespace()
            data_item.non_tensor_batch = {
                "raw_prompt": [{"content": observation_text}],
                "profile_id": profile,
            }
            data_item.batch = {"responses": action_text}
            return float(metric.score(data_item))
        except Exception:
            return 0.0

    def _compute_proagent_dense_reward(
        self,
        sparse_reward: float,
        env_info: Dict[str, Any],
        valid: Any,
        joint_action: Any,
        is_first_internal_step: bool,
    ) -> float:
        """
        Compute scalar dense reward for *one env* at one executed low-level env.step.

        Notes:
        - OvercookedEnv returns sparse reward as `reward`, and shaped reward as env_info['shaped_r'].
        - We keep everything scalar per env (what your trainer expects).
        """
        cfg = self._reward_cfg

        # Base sparse reward (delivery reward etc.)
        dense = float(sparse_reward) if sparse_reward is not None else 0.0

        # Add env-shaped reward (if available)
        shaped_r = 0.0
        if cfg.get("include_shaped_r", True) and isinstance(env_info, dict):
            try:
                shaped_r = float(env_info.get("shaped_r", 0.0) or 0.0)
            except Exception:
                shaped_r = 0.0
            dense += float(cfg.get("shaped_coef", 1.0)) * shaped_r

        # Step penalty (applied per executed low-level step)
        dense += float(cfg.get("step_penalty", 0.0) or 0.0)

        # STAY penalty for agent0 if we can infer it
        # Overcooked Action.STAY is the tuple (0, 0)
        try:
            if isinstance(joint_action, (list, tuple)) and len(joint_action) > 0 and joint_action[0] == (0, 0):
                dense += float(cfg.get("stay_penalty", 0.0) or 0.0)
        except Exception:
            pass

        # Projection invalid penalty: apply only once at the first internal step (cnt == 0)
        try:
            is_valid = bool(valid)
        except Exception:
            is_valid = True
        if is_first_internal_step and not is_valid:
            dense += float(cfg.get("invalid_action_penalty", 0.0) or 0.0)

        # Scale and clip
        dense *= float(cfg.get("reward_scale", 1.0) or 1.0)

        clip = cfg.get("reward_clip", None)
        if clip is not None and isinstance(clip, (list, tuple)) and len(clip) == 2:
            try:
                lo, hi = float(clip[0]), float(clip[1])
                if lo > hi:
                    lo, hi = hi, lo
                dense = float(np.clip(dense, lo, hi))
            except Exception:
                pass

        # Optional debug breakdown (attach to env_info to make tuning easy)
        if isinstance(env_info, dict) and cfg.get("debug_reward_breakdown", False):
            env_info["reward_breakdown"] = {
                "sparse": float(sparse_reward) if sparse_reward is not None else 0.0,
                "shaped": float(shaped_r),
                "dense": float(dense),
                "cfg": {
                    "shaped_coef": float(cfg.get("shaped_coef", 1.0)),
                    "step_penalty": float(cfg.get("step_penalty", 0.0) or 0.0),
                    "stay_penalty": float(cfg.get("stay_penalty", 0.0) or 0.0),
                    "invalid_action_penalty": float(cfg.get("invalid_action_penalty", 0.0) or 0.0),
                    "reward_scale": float(cfg.get("reward_scale", 1.0) or 1.0),
                    "reward_clip": clip,
                },
            }

        return float(dense)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        self.pre_text_obs = obs
        #breakpoint()
        # Get config for agent1 (API agent)
        base_url = getattr(self.config.env.proagent, 'base_url', None)
        lm_id = getattr(self.config.env.proagent, 'lm_id', None)
        sampling_parameters_raw = getattr(self.config.env.proagent, 'sampling_parameters', None)
        
        # Convert sampling_parameters to dict if it's an OmegaConf object
        if sampling_parameters_raw is not None:
            try:
                from omegaconf import OmegaConf
                if hasattr(sampling_parameters_raw, '_content'):
                    sampling_parameters = OmegaConf.to_container(sampling_parameters_raw, resolve=True)
                else:
                    sampling_parameters = sampling_parameters_raw
            except:
                sampling_parameters = sampling_parameters_raw
        else:
            sampling_parameters = {}
        
        # Import prompt template
        from agent_system.environments.prompts.proagent import PROAGENT_ACTION_TEMPLATE
        prompt_template = PROAGENT_ACTION_TEMPLATE
        
        # Get batch client configuration
        batch_max_size = getattr(self.config.env.proagent, 'batch_max_size', 8)
        batch_max_wait_time = getattr(self.config.env.proagent, 'batch_max_wait_time', 0.5)
        use_batch_client = getattr(self.config.env.proagent, 'use_batch_client', True)
        
        # Reset memory with agent configuration
        self.memory.reset(
            batch_size=len(obs),
            obs=obs,
            infos=infos,
            base_url=base_url,
            lm_id=lm_id,
            prompt_template=prompt_template,
            sampling_parameters=sampling_parameters,
            p0=self.p0,
            p1=self.p1,
            batch_max_size=batch_max_size,
            batch_max_wait_time=batch_max_wait_time,
            use_batch_client=use_batch_client,
            group_n=getattr(self.envs, "group_n", 1),
            seed=getattr(self.config.env, "seed", 0),
        )
        
        observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': self.build_anchor_obs()
        }
        self.steps = [0] * len(obs)
        return observations, infos
    
    def parse_action_text(self, action_text):
        # result=""
        Analysis=""
        message=""
        action="So"
        import re
        think_pattern = r'<think>\s*(.+?)\s*</think>'
        think_match = re.findall(think_pattern, action_text, re.DOTALL)
        if think_match:
            Analysis= think_match[-1].strip()
        # Extract message from <message> tags
        message_pattern = r'<message>\s*(.+?)\s*</message>'
        message_match = re.findall(message_pattern, action_text, re.DOTALL)
        if message_match:
            message= message_match[-1].strip()
        action_pattern = r'<action>\s*(.+?)\s*</action>'
        action_match = re.findall(action_pattern, action_text, re.DOTALL)
        if action_match:
            action = action_match[-1].strip()
        if "message" in action:
            action=action+"<"+message+">"
        result=Analysis+action
        return result

    def step(self, text_actions: List[str]):
        """
        Execute high-level actions (ML actions) for agent0 until completion.
        Agent0 and agent1 execute synchronously, but when agent0 completes its ML action,
        the environment stops even if agent1 hasn't finished.
        """
        #breakpoint()
        all_done = False
        action_infos = []
        rewards = [0.0] * self.envs.num_processes
        step_info = None
        cnt = 0
        available_actions = self.build_available_actions()
        interactivity_scores = None
        
        if self.debug:
            print(f"player0 : Text actions: {text_actions}")
        
        # Loop until all environments' agent0 complete their high-level actions
        while not all_done:
            states = self.envs.get_state()
            obs = self.envs.get_observations()
            
            # Project text actions to environment actions
            # On first call (cnt == 0), parse ML actions from text_actions
            # On subsequent calls (cnt > 0), agent0 continues executing current ML action
            actions, valids, action_infos = self.projection_f(text_actions, self.memory, obs, None, action_infos, cnt, available_actions, debug=self.debug)
            
            # Compute interactivity scores on first step (cnt == 0) after getting action_infos
            if cnt == 0:
                if interactivity_scores is None:
                    interactivity_scores = [0.0] * self.envs.num_processes
                    try:
                        obs_texts = self.memory.build_text_obs(states)
                    except Exception:
                        try:
                            obs_texts = [str(s) for s in states] if isinstance(states, list) else [str(states)]
                        except Exception:
                            obs_texts = []
                    
                    metric = self._get_interactivity_metric()
                    for idx in range(self.envs.num_processes):
                        obs_text = obs_texts[idx] if idx < len(obs_texts) else "Unknown state"
                        
                        # 直接使用 rollout 解码出的 text_actions 作为交互文本
                        act_text = text_actions[idx] if isinstance(text_actions, list) and idx < len(text_actions) else text_actions
                        act_text=self.parse_action_text(act_text)
                        print(f"env{idx}: {act_text}")
                        
                        interactivity_scores[idx] = self._compute_interactivity_score(metric, obs_text, act_text, env_idx=idx)
            #breakpoint()

            # Snapshot completion status BEFORE this low-level env.step.
            # This ensures we still count rewards from the step that *completes* an ML action
            # (completion is detected after the step based on the new state).
            completed_before_step = [False] * self.envs.num_processes
            for i in range(self.envs.num_processes):
                try:
                    if i < len(action_infos) and isinstance(action_infos[i], dict) and action_infos[i].get(0) is not None:
                        info_p0 = action_infos[i][0]
                        if isinstance(info_p0, dict) and info_p0.get("ml_action_completed", False):
                            completed_before_step[i] = True
                except Exception:
                    completed_before_step[i] = False
            
            # Execute actions
            # 传递 action_infos 和 step_info 给 envs.step()，以便已完成的环境可以跳过执行
            if step_info is None:
                # First step:
                # 这里也要提供一个“初始 step_info”，这样当某些环境在本轮就被标记为 ml_action_completed 时，
                # worker 可以直接返回该 step_info，从而真正做到“不执行 env.step()、timestep 不变”
                init_obs = obs
                init_rewards = [0.0] * self.envs.num_processes
                init_dones = [False] * self.envs.num_processes
                init_infos = [{} for _ in range(self.envs.num_processes)]
                init_step_info = (init_obs, init_rewards, init_dones, init_infos)

                next_obs, step_rewards, dones, infos = self.envs.step(actions, action_infos, init_step_info)
                step_info = (next_obs, step_rewards, dones, infos)
            else:
                # Subsequent steps: continue from previous state
                # 传递 step_info，让已完成的环境直接返回，不执行实际的 step
                next_obs, step_rewards, dones, infos = self.envs.step(actions, action_infos, step_info)
                step_info = (next_obs, step_rewards, dones, infos)

            # Compute dense step rewards (sparse + env shaped + extras) per env.
            # Important: only meaningful for envs that actually executed env.step (i.e., have a non-empty action).
            dense_step_rewards = [0.0] * self.envs.num_processes
            for i in range(self.envs.num_processes):
                try:
                    joint_action = actions[i] if isinstance(actions, list) and i < len(actions) else None
                    has_any_action = (
                        joint_action is not None
                        and isinstance(joint_action, (list, tuple))
                        and any(a is not None for a in joint_action)
                    )
                    if not has_any_action:
                        dense_step_rewards[i] = 0.0
                        continue
                    dense_step_rewards[i] = self._compute_proagent_dense_reward(
                        sparse_reward=step_rewards[i] if i < len(step_rewards) else 0.0,
                        env_info=infos[i] if i < len(infos) else {},
                        valid=valids[i] if i < len(valids) else 1,
                        joint_action=joint_action,
                        is_first_internal_step=(cnt == 0),
                    )
                except Exception:
                    dense_step_rewards[i] = step_rewards[i] if i < len(step_rewards) else 0.0

            # Debug print (per low-level step)
            if self._reward_cfg.get("debug_print_each_low_step", False):
                try:
                    every_n = int(self._reward_cfg.get("debug_print_every_n", 1) or 1)
                    if every_n <= 0:
                        every_n = 1
                    if (cnt % every_n) == 0:
                        max_envs = int(self._reward_cfg.get("debug_print_max_envs", 3) or 3)
                        max_envs = max(1, max_envs)
                        n = min(self.envs.num_processes, max_envs)
                        lines = [f"[ProAgentEnvManager] internal_cnt={cnt}"]
                        for i in range(n):
                            info_i = infos[i] if i < len(infos) else {}
                            shaped_i = info_i.get("shaped_r", 0.0) if isinstance(info_i, dict) else 0.0
                            ja = actions[i] if isinstance(actions, list) and i < len(actions) else None
                            lines.append(
                                f"  env{i}: sparse={float(step_rewards[i]):.3f} shaped={float(shaped_i):.3f} "
                                f"dense={float(dense_step_rewards[i]):.3f} acc={float(rewards[i]):.3f} "
                                f"done={bool(dones[i])} valid={bool(valids[i]) if i < len(valids) else True} action={ja}"
                            )
                        if self.debug:
                            print("\n".join(lines), flush=True)
                except Exception:
                    pass
            
            # 执行 step 后，检查哪些环境的 agent0 完成了 ML 动作（基于新的 state）
            # 这是最准确的判断时机，因为完成判断应该基于执行后的状态
            for i in range(self.envs.num_processes):
                # 跳过已经标记为完成的环境
                if i < len(action_infos) and action_infos[i].get(0) is not None:
                    info_p0 = action_infos[i][0]
                    if info_p0.get('ml_action_completed', False):
                        continue  # 已经完成，跳过检查

                # 检查 agent0 的 ML 动作是否完成（通过 Ray actor）
                if hasattr(self.memory, 'check_ml_action_done'):
                    try:
                        # 基于执行后的 state 检查完成
                        state = next_obs[i] if isinstance(next_obs, list) else next_obs
                        if state is not None:
                            ml_action_done = self.memory.check_ml_action_done(i, state)
                            if self.debug:
                                print(f"[env {i}] check ml_action_done: {ml_action_done}")
                            
                            if ml_action_done:
                                # 执行这一步后，ML 动作完成了
                                # 标记完成，下一次循环时会跳过执行
                                if action_infos[i].get(0) is None:
                                    action_infos[i][0] = {}
                                action_infos[i][0]['ml_action_completed'] = True
                                # 从当前 info 中获取 ml_action（agent_pair_worker 已经设置了）
                                if 'ml_action' not in action_infos[i][0]:
                                    action_infos[i][0]['ml_action'] = None
                                if 'source' not in action_infos[i][0]:
                                    action_infos[i][0]['source'] = 'RL'
                    except Exception as e:
                        if self.debug:
                            print(f"Error checking ML action completion for env {i} after step: {e}")
                        import traceback
                        traceback.print_exc()
            

            
            # Accumulate rewards (do NOT accumulate for envs where agent0 ML action already completed).
            # NOTE: We still want to count terminal-step rewards, but avoid double counting when workers return cached step_info.
            for i in range(len(rewards)):
                # Only skip envs that were already completed BEFORE this step.
                # (If an ML action becomes completed AFTER this step, we still want to count this step's reward.)
                is_completed = completed_before_step[i] if i < len(completed_before_step) else False
                
                # Only accumulate if:
                # - agent0 has NOT completed ML action
                # - this env actually executed env.step (has any low-level action)
                try:
                    joint_action = actions[i] if isinstance(actions, list) and i < len(actions) else None
                    has_any_action = (
                        joint_action is not None
                        and isinstance(joint_action, (list, tuple))
                        and any(a is not None for a in joint_action)
                    )
                except Exception:
                    has_any_action = True

                if not is_completed and has_any_action:
                    rewards[i] += dense_step_rewards[i]*0.1
                    if cnt == 0:
                        try:
                            # 添加 interactivity score
                            rewards[i] += float(interactivity_scores[i]) if interactivity_scores is not None else 0.0
                            
                            # 添加 set_message reward
                            if isinstance(action_infos, list) and i < len(action_infos):
                                info_dict = action_infos[i]
                                if isinstance(info_dict, dict) and 0 in info_dict:
                                    p0_info = info_dict[0]
                                    if isinstance(p0_info, dict):
                                        ml_action = p0_info.get("ml_action", "")
                                        if ml_action and isinstance(ml_action, str) and "set_message" in ml_action.lower():
                                            # 添加 set_message reward
                                            set_message_reward = float(self._reward_cfg.get("set_message_reward", 0.01))
                                            rewards[i] += set_message_reward
                                            if self.debug:
                                                print(f"[env {i}] Added set_message reward: {set_message_reward}")
                        except Exception:
                            pass
            
            agent0_completed = []
            for i in range(self.envs.num_processes):
                if dones[i]:
                    # Episode 结束，agent0 视为完成
                    agent0_completed.append(True)
                else:
                    # 检查 action_infos 中的完成标志（这是最准确的判断方式）
                    try:
                        if i < len(action_infos) and action_infos[i].get(0) is not None:
                            info_p0 = action_infos[i][0]
                            if info_p0.get('ml_action_completed', False):
                                agent0_completed.append(True)
                            else:
                                # 未完成
                                agent0_completed.append(False)
                        else:
                            # 没有 action_infos，视为未完成（第一次调用时）
                            agent0_completed.append(False)
                    except Exception as e:
                        if self.debug:
                            print(f"Error checking ML action completion for env {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        # 出错时视为未完成，避免误判
                        agent0_completed.append(False)
            
            # All environments' agent0 have completed their ML actions
            all_done = all(agent0_completed)
            
            # Update steps for completed episodes
            for i in range(self.envs.num_processes):
                if dones[i] and self.steps[i] == 0:
                    try:
                        if i < len(states) and states[i] is not None:
                            self.steps[i] = states[i].timestep
                        else:
                            self.steps[i] = infos[i].get('timestep', len(self.memory[i]) if hasattr(self.memory, '__getitem__') and i < len(self.memory) else 0)
                    except:
                        self.steps[i] = len(self.memory[i]) if hasattr(self.memory, '__getitem__') and i < len(self.memory) else 0
            
            cnt += 1
            
            # Safety: prevent infinite loops
            if cnt > 1000:
                if self.debug:
                    print(f"Warning: Step loop exceeded 1000 iterations, forcing completion")
                all_done = True
        
        # Extract final results
        next_obs, _, dones, infos = step_info
        
        # Update memory
        try:
            states = self.envs.get_state()
            self.memory.store({
                'text_obs': [self._state_to_text(s) if s is not None else "Unknown state" for s in states],
                'action': [str(a) for a in actions]
            })
        except:
            pass  # Memory store is optional
        
        self.pre_text_obs = next_obs
        
        # Build next observations
        next_observations = {
            'text': self.build_text_obs(),
            'image': None,
            'anchor': self.build_anchor_obs()
        }
        
        # Add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['step'] = self.steps[i]
            # Add 'won' key for success evaluation (required by base.py success_evaluator)
            # For proagent (Overcooked), 'won' represents episode score:
            # Overcooked is a fixed-horizon game where score is based on cumulative rewards
            # - If done: use total episode reward (ep_sparse_r + ep_shaped_r) as score
            # - If not done: use current cumulative reward as progress indicator
            if hasattr(self, 'config') and hasattr(self.config.env, 'env_name') and 'proagent' in self.config.env.env_name.lower():
                if dones[i]:
                    # Episode ended: use total episode reward as score
                    # OvercookedEnv returns info with 'episode' dict containing:
                    #   - ep_sparse_r: cumulative sparse rewards (soup deliveries, default 20 per delivery)
                    #   - ep_shaped_r: cumulative shaped rewards (other rewards like proximity to goals)
                    if 'episode' in info:
                        ep_info = info['episode']
                        # Total score = sparse rewards + shaped rewards
                        ep_sparse_r = ep_info.get('ep_sparse_r', 0.0)
                        ep_shaped_r = ep_info.get('ep_shaped_r', 0.0)
                        total_score = ep_sparse_r + ep_shaped_r
                        info['won'] = float(total_score)
                    else:
                        # Fallback: use current reward if episode info not available
                        # This shouldn't happen, but handle gracefully
                        info['won'] = float(rewards[i]) if rewards[i] > 0 else 0.0
                else:
                    # Episode ongoing: use accumulated reward as progress indicator
                    info['won'] = float(rewards[i])
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        # Debug print right before returning to trainer
        if self.debug and self._reward_cfg.get("debug_print_return", False):
            try:
                r_np = np.array(rewards, dtype=float)
                print(
                    f"[ProAgentEnvManager] return rewards: shape={r_np.shape} "
                    f"min={float(np.min(r_np)):.3f} mean={float(np.mean(r_np)):.3f} max={float(np.max(r_np)):.3f}",
                    flush=True,
                )
                # show a small sample
                max_envs = int(self._reward_cfg.get("debug_print_max_envs", 3) or 3)
                n = min(len(infos), max_envs)
                for i in range(n):
                    info_i = infos[i] if i < len(infos) else {}
                    shaped_i = info_i.get("shaped_r", 0.0) if isinstance(info_i, dict) else 0.0
                    rb = info_i.get("reward_breakdown") if isinstance(info_i, dict) else None
                    print(f"  env{i}: reward={float(r_np[i]):.3f} shaped_r={shaped_i} breakdown={rb}", flush=True)
            except Exception:
                pass
        
        return next_observations, rewards, dones, infos

    def build_text_obs(self) -> List[str]:
        """Build text observations from game states"""
        obs = self.envs.get_observations()
        states = self.envs.get_state()
        
        # Use memory to build text observations
        total_obs = self.memory.build_text_obs(states)
        text_obs = []
        for item in total_obs:
            if isinstance(item, tuple):
                text_obs.append(item[0])
            else:
                text_obs.append(item)
        return text_obs

    def build_available_actions(self) -> List[str]:
        """Build available actions list"""
        states = self.envs.get_state()
        available_actions = []
        
        # For Overcooked, actions are always the same
        actions = ["NORTH", "SOUTH", "EAST", "WEST", "INTERACT", "STAY"]
        action_str = "\n".join([f"{chr(ord('A')+j)}. {action}" for j, action in enumerate(actions)])
        
        for _ in states:
            available_actions.append(action_str)
        
        return available_actions
    
    def build_anchor_obs(self) -> List[Dict]:
        """Build anchor observations"""
        states = self.envs.get_state()
        # Use memory to build anchor observations
        anchor_obs = self.memory.build_anchor_obs(states)
        return anchor_obs
    
    def _state_to_text(self, state):
        """Convert Overcooked state to text"""
        if state is None:
            return "Initial state"
        
        try:
            # Use the state string representation
            if hasattr(state, 'mdp') and hasattr(state.mdp, 'state_string'):
                state_text = state.mdp.state_string(state).replace('ø', 'o')
                return state_text
            else:
                return str(state)
        except Exception as e:
            return f"State: {str(state)[:100]}"



def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "cwah" in config.env.env_name.lower():
        from agent_system.environments.env_package.cwah import build_cwah_envs, cwah_projection
        env_kwargs = {
            'max_episode_length': 250, 
            'env_task_set': pickle.load(open("agent_system/environments/env_package/cwah/cwah/dataset/test_env_set_help.pik", 'rb')),
            'env_task_set_index': {"train": list(set(range(50)) - set((0, 5, 10, 16, 20, 26, 30, 32, 40, 49))), "val": [0, 5, 10, 16, 20, 26, 30, 32, 40, 49]}, 
            'obs_type': 'partial',
            'use_editor': False,
            'executable_args': {
                'file_name': "agent_system/environments/env_package/cwah/executable/linux_exec.v2.3.0.x86_64",
                'no_graphics': True,
                }
            }
        _envs = build_cwah_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker, env_kwargs=env_kwargs)
        _val_envs = build_cwah_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker, env_kwargs=env_kwargs)
        
        projection_f = partial(cwah_projection)
        envs = CwahEnvironmentManager(_envs, projection_f, config)
        val_envs = CwahEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "proagent" in config.env.env_name.lower():
        from agent_system.environments.env_package.proagent import build_proagent_envs, proagent_projection
        # Reward shaping is enabled by default; override via config.env.proagent.enable_reward_shaping / rew_shaping_params
        try:
            rew_shaping_params = config.env.proagent.get('rew_shaping_params', None)
        except Exception:
            rew_shaping_params = None

        # Convert OmegaConf dicts to plain dicts (if needed)
        if rew_shaping_params is not None and not isinstance(rew_shaping_params, dict):
            try:
                rew_shaping_params = OmegaConf.to_container(rew_shaping_params, resolve=True)
            except Exception:
                rew_shaping_params = None

        # Support both single layout and multi-layout configs
        layouts_config = config.env.proagent.get('layouts', None)
        single_layout = config.env.proagent.get('layout', None)
        
        # Convert OmegaConf list to plain list if needed
        if layouts_config is not None:
            try:
                if hasattr(layouts_config, '_content'):
                    layouts_config = OmegaConf.to_container(layouts_config, resolve=True)
            except Exception:
                pass
        
        env_kwargs = {
            'layouts': layouts_config,  # Multi-layout mode (takes precedence)
            'layout': single_layout,    # Single layout mode (backward compatible)
            'horizon': config.env.proagent.get('horizon', 400),
            'enable_reward_shaping': config.env.proagent.get('enable_reward_shaping', True),
            'rew_shaping_params': rew_shaping_params,
            'debug': config.env.proagent.get('debug', False),
        }
        _envs = build_proagent_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker, env_kwargs=env_kwargs)
        _val_envs = build_proagent_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker, env_kwargs=env_kwargs)
        
        projection_f = partial(proagent_projection)
        envs = ProAgentEnvironmentManager(_envs, projection_f, config)
        val_envs = ProAgentEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)