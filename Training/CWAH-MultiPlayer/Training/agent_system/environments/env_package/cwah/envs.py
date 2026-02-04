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

from agent_system.environments.env_package.cwah.cwah.envs.unity_environment import UnityEnvironment

def env_fn(seed, env_id, base_port, is_train, env_kwargs):
    return UnityEnvironment(num_agents=2,
                            max_episode_length=env_kwargs["max_episode_length"],
                            port_id=env_id,
                            env_task_set=env_kwargs["env_task_set"],
                            agent_goals=['LLM', 'LLM'],
                            observation_types=[env_kwargs["obs_type"], env_kwargs["obs_type"]],
                            use_editor=env_kwargs["use_editor"],
                            executable_args=env_kwargs["executable_args"],
                            base_port=base_port,
                            seed=seed)

class CwahWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, seed, env_id, base_port, is_train, env_kwargs):
        self.env = env_fn(seed, env_id, base_port, is_train, env_kwargs)  # Each worker holds only one sub-environment
    
    def step(self, action, infos, step_info = None):
        """Execute a step in the environment"""
        if len(action) == 0 or (action[0] is None and infos[0]['plan'] != "[wait]"):
            return step_info
        else:
            obs, scores, dones, infos, messages = self.env.step(action)
            return obs, scores, dones, infos, messages
    
    def reset(self, task_id):
        """Reset the environment"""
        obs = self.env.reset(task_id=task_id)
        infos = {
            'all_containers_name': self.env.all_containers_name,
            'all_goal_objects_name': self.env.all_goal_objects_name,
            'all_room_name': self.env.all_room_name,
            'room_info': self.env.room_info,
            'goal_spec': self.env.goal_spec,
        }
        return obs, infos
    
    def get_observations(self):
        obs = self.env.get_observations()
        return obs

    def get_goal(self):
        goal_spec = [self.env.get_goal(self.env.task_goal[0], self.env.agent_goals[0]), self.env.get_goal(self.env.task_goal[1], self.env.agent_goals[1])]
        return goal_spec
    
    def get_step(self):
        steps = self.env.get_step()
        return steps
    

class CwahEnvs(gym.Env):
    def __init__(self, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        self.env_kwargs = env_kwargs

        random.seed(seed)
        np.random.seed(seed)

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(CwahWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed, i, np.random.randint(11000, 13000), is_train, env_kwargs)
            self.workers.append(worker)

    def step(self, actions, infos, step_info):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            if len(actions[i]) == 0 or (actions[i][0] is None and infos[i][0]['plan'] != "[wait]"):
                future = worker.step.remote(actions[i], infos[i], (step_info[0][i], step_info[1][i], step_info[2][i], step_info[3][i], step_info[4][i]))
            else:
                future = worker.step.remote(actions[i], infos[i])
            futures.append(future)

        # Collect results
        obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        message_list = []

        results = ray.get(futures)

        for i, (obs, reward, dones, info, message) in enumerate(results):
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(dones)
            info_list.append(info)
            message_list.append(message)
        return obs_list, rewards_list, dones_list, info_list, message_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        obs_list = []
        infos_list = []

        # randomly generate self.env_num seeds
        futures = []
        if self.is_train:
            task_id = np.random.choice(self.env_kwargs["env_task_set_index"]["train"], size=self.env_num)
        else:
            task_id = np.random.choice(self.env_kwargs["env_task_set_index"]["val"], size=self.env_num)

        # repeat the seeds for each group
        task_id = np.repeat(task_id, self.group_n)
        task_id = task_id.tolist()

        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(task_id=task_id[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            infos_list.append(info)

        return obs_list, infos_list
    
    def get_observations(self):

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.get_observations.remote()
            futures.append(future)

        # Collect results
        obs_list = []

        results = ray.get(futures)
        for i, obs in enumerate(results):
            obs_list.append(obs)
        return obs_list
    
    def get_goal(self):

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.get_goal.remote()
            futures.append(future)

        # Collect results
        goal_list = []

        results = ray.get(futures)
        for i, goal in enumerate(results):
            goal_list.append(goal)
        return goal_list
    
    def get_step(self):

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.get_step.remote()
            futures.append(future)

        # Collect results
        step_list = []

        results = ray.get(futures)
        for i, step in enumerate(results):
            step_list.append(step)
        return step_list

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_cwah_envs(seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return CwahEnvs(seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)