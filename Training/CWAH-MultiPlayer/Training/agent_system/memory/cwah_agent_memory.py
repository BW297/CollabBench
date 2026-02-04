from typing import List, Dict, Any, Tuple
from .base import BaseMemory

import ray
import random

from openai import OpenAI
import json
import re


class LLM:
    def __init__(self,
                 base_url,
                 lm_id,
                 prompt_template,
                 sampling_parameters,
                 agent_id
                 ):
        self.goal_desc = None
        self.goal_location_with_r = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_name = "Alice" if agent_id == 2 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 2 else "he"
        self.debug = sampling_parameters.debug
        self.goal_location = None
        self.goal_location_id = None
        self.roomname2id = {}
        self.rooms = []
        self.communication = True
        self.cot = True
        self.chat = True
        self.total_cost = 0
        self.single = False
        # self.prompt_template = prompt_template.format(
        #     AGENT_NAME = self.agent_name,
        #     OPPO_NAME = self.oppo_name
        # )
        self.prompt_template = prompt_template
        # if agent_id == 2:
        #     self.prompt_template = self.prompt_template.replace('$ASSIGNED_PERSONALITY$', 'High Neuroticism')
        #     self.prompt_template = self.prompt_template.replace('$BEHAVIOR_STYLE$', 'Becomes anxious when things go wrong, for instance repeatedly [send_message] \"Are you sure you closed the fridge? Please confirm\" while wasting time near the object, or hesitating between Walk towards bread and Walk towards apple instead of committing, which reduces overall efficiency.')
        self.base_url = base_url                           
        self.lm_id = lm_id
        self.api_key = "None"
        self.sampling_params = {
            "max_tokens": sampling_parameters.max_tokens,
            "temperature": sampling_parameters.t,
            "top_p": sampling_parameters.top_p,
            "n": sampling_parameters.n,
        }
            
        def lm_engine(lm_id):
            def _generate(prompt, sampling_params):
                client = OpenAI(api_key = self.api_key, base_url = self.base_url)
                response = client.chat.completions.create(
                    model=lm_id, messages=prompt, **sampling_params
                )
                # print(json.dumps(response, indent=4))
                if self.debug:
                    with open(f"LLM/chat_raw.json", 'a') as f:
                        f.write(json.dumps(response, indent=4))
                        f.write('\n')
                # generated_samples = [response.choices[i].message.content for i in
                #                         range(sampling_params['n'])]
                return response.choices[0].message.content, 0
            return _generate

        self.generator = lm_engine(self.lm_id)


    def reset(self, rooms_name, roomname2id, goal_location, unsatisfied):
        self.rooms = rooms_name
        self.roomname2id = roomname2id
        self.goal_location = goal_location
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)


    def goal2description(self, goals, goal_location_room):  # {predicate: count}
        # print(goals)
        map_rel_to_pred = {
            'inside': 'into',
            'on': 'onto',
        }
        s = "Find and put "
        r = None
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split('_')
            count = vl
            if count == 0:
                continue
            if relation == 'holds':
                continue
                # s += f"Alice holds a book, "
            elif relation == 'sit':
                continue
                # s += f"Alice sits in {obj2}, "
            else:
                s += f"{count} {obj1}{'s' if count > 1 else ''}, "
                r = relation
        if r is None:
            return "None."

        s = s[:-2] + f" {map_rel_to_pred[r]} the {self.goal_location}."
        # if type(goal_location_room) is not list:
        #     s += f" in the {goal_location_room}."
        # else:
        #     ss = ' or '.join([f'{room}' for room in goal_location_room])
        #     s += f", which may be in the {ss}."
        return s, f"{map_rel_to_pred[r]} the {self.goal_location}"


    # def get_obj(self, obs, text, k=1):
    #     id2node = {node['id']: node for node in obs['nodes']}
    #     cnt = 0
    #     for x, node in id2node.items():
    #         if f'({x})' in text:
    #             cnt += 1
    #             if cnt != k: continue
    #             return f"<{node['class_name']}> ({x})"
    #     print("WARNING! No object correctly parsed!!! Random choose one")
    #     x, node = random.choice(list(id2node.items()))
    #     return f"<{node['class_name']}> ({x})"
    #
    #
    # def get_action(self, obs, text):
    #     if '[open]' in text or '[close]' in text or '[grab]' in text or '[walktowards]' in text:
    #         return f"[{text.split(']')[0].split('[')[-1]}] {self.get_obj(obs, text)}"
    #     elif 'putback' in text or 'putin' in text:
    #         obj1 = self.get_obj(obs, text)
    #         obj2 = self.get_obj(obs, text, 2)
    #         return f"[{text.split(']')[0].split('[')[-1]}] {obj1} {obj2}"

    def parse_answer(self, available_actions, text):
        original_str = text.lower()
        text = text.lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)
        if start_idx == -1 or end_idx == -1:
            # If we can't find a valid <action>...</action> block, mark as invalid
            text = text[-30:]  # 0 is invalid action for Sokoban
        else:
            # Extract just the content between the tags
            text = text[start_idx + len(start_tag):end_idx].strip().lower()

        if "[send_message]" in text:
            start_tag = "<message>"
            end_tag = "</message>"
            start_idx = original_str.find(start_tag)
            end_idx = original_str.find(end_tag)
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                return random.choice(available_actions) # 0 is invalid action for Sokoban
            else:
                # Extract just the content between the tags
                return f"[send_message] <{original_str[start_idx + len(start_tag):end_idx].strip()}>"
            
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
                return action
        print("WARNING! Fuzzy match!")
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act, name, id = action.split(' ')
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        print(text)
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions)


    def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
        sss = {}
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                extra_obj = self.goal_location
            if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
                sss[room] = f"The {room} is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"{extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"<{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
                    s_con = f"unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += 'nothing'
                if room_explored is not None and not room_explored[room]:
                    s += ' yet'
            elif s_obj != "" and s_con != "":
                s += s_obj + ', and ' + s_con
            else:
                s += s_obj + s_con
            sss[room] = s

        if len(satisfied) == 0:
            s = ""
        else:
            s = f"{'I' if self.single else 'We'}'ve already found and put "
            s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
            s += ' ' + self.goal_location_with_r + '. '

        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
        ### opponent modeling
        if not self.single:
            ss = ""
            if len(opponent_grabbed_objects) == 0:
                ss += "nothing. "
            else:
                ss += f"<{opponent_grabbed_objects[0]['class_name']}> ({opponent_grabbed_objects[0]['id']}). "
                if len(opponent_grabbed_objects) == 2:
                    ss = ss[:-2] + f" and <{opponent_grabbed_objects[1]['class_name']}> ({opponent_grabbed_objects[1]['id']}). "
            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == current_room['class_name']:
                s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "

        return s


    def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, room_explored):
        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        [send_message] <"">
        """
        available_plans = []
        if self.communication:
            available_plans.append(f"[send_message] <>")
        for room in self.rooms:
            if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
                continue
            available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        if len(grabbed_objects) > 0:
            available_plans.append(f"[goput] {self.goal_location}")
        # available_plans = available_plans[:2]
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

            
    def run(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None, text_action=None):
        info = {}
        # goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history)
        # prompt = self.prompt_template.format(GOAL = self.goal_desc)
        # prompt = prompt.format(PROGRESS = progress_desc)
        # prompt = prompt.format(ACTION_HISTORY = action_history_desc)
        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers, ungrabbed_objects, room_explored)
        prompt = self.prompt_template.format(
            AGENT_NAME = self.agent_name,
            OPPO_NAME = self.oppo_name,
            GOAL = self.goal_desc,
            PROGRESS = progress_desc,
            ACTION_HISTORY = action_history_desc,
            DIALOGUE_HISTORY = dialogue_history_desc,
            AVAILABLE_ACTIONS = available_plans
        )
        message = None

        # if self.communication:
        #     prompt = prompt.format(DIALOGUE_HISTORY = dialogue_history_desc)
            # if not action_history[-1].startswith('[send_message]'):
            #     gen_prompt = self.generator_prompt_template.replace('$GOAL$', self.goal_desc)
            #     gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
            #     gen_prompt = gen_prompt.replace('$ACTION_HISTORY$', action_history_desc)
            #     gen_prompt = gen_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
            #     gen_prompt = gen_prompt + f"\n{self.agent_name}:"
            #     chat_prompt = [{"role": "user", "content": gen_prompt}]
            #     outputs, usage = self.generator(chat_prompt if self.chat else gen_prompt, self.sampling_params)
            #     self.total_cost += usage
            #     message = outputs[0]
            #     info['message_generator_prompt'] = gen_prompt
            #     info['message_generator_outputs'] = outputs
            #     info['message_generator_usage'] = usage
            #     if self.debug:
            #         print(f"message_generator_prompt:\n{gen_prompt}")
            #         print(f"message_generator_outputs:\n{message}")
        if num == 1:
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                     "plan": None})
            return plan, info

        if self.cot:
            # prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            if text_action is None:
                output, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            else:
                output = text_action
                usage = 0
            self.total_cost += usage
            info['cot_outputs'] = output
            info['cot_usage'] = usage
            if self.debug:
                print(f"cot_output:\n{output}")
            # chat_prompt = [{"role": "user", "content": prompt},
            #                {"role": "assistant", "content": output},
            #                {"role": "user", "content": "Answer with only one best next action. So the answer is"}]
            # normal_prompt = prompt + output + ' So the answer is'
            # outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            # output = outputs[0]
            # self.total_cost += usage
            info['output_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
                print(f"total cost: {self.total_cost}")
        else:
            if self.debug:
                print(f"base_prompt:\n{prompt}")
            outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt, self.sampling_params)
            output = outputs[0]
            info['cot_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
        plan = self.parse_answer(available_plans_list, output)
        if self.debug:
            print(f"plan: {plan}\n")
        info.update({"num_available_actions": num,
                     "prompts": prompt,
                     "outputs": [output],
                     "plan": plan,
                     "total_cost": self.total_cost})
        return plan, info
    
    def get_text_obs(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history)
        # dialogue_history_desc = '\n'.join(dialogue_history)
        # prompt = self.prompt_template.format(GOAL = self.goal_desc)
        # prompt = prompt.format(PROGRESS = progress_desc)
        # prompt = prompt.format(ACTION_HISTORY = action_history_desc)
        # if self.communication:
        #     prompt = prompt.format(DIALOGUE_HISTORY = dialogue_history_desc)
        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers, ungrabbed_objects, room_explored)
        prompt = self.prompt_template.format(
            AGENT_NAME = self.agent_name,
            OPPO_NAME = self.oppo_name,
            GOAL = self.goal_desc,
            PROGRESS = progress_desc,
            ACTION_HISTORY = action_history_desc,
            DIALOGUE_HISTORY = dialogue_history_desc,
            AVAILABLE_ACTIONS = available_plans
        )
        # if self.cot:
        #     prompt = prompt + " Let's think step by step."
        return (prompt, available_plans_list)



class cwah_agent:
    """
    LLM agent class
    """
    def __init__(self, agent_id, base_url, lm_id, prompt_template, sampling_parameters):
        self.debug = False
        self.agent_type = 'LLM'
        self.agent_names = ["Zero", "Alice", "Bob"]
        self.agent_id = agent_id
        self.opponent_agent_id = 3 - agent_id
        self.lm_id = lm_id
        self.LLM = LLM(base_url, self.lm_id, prompt_template, sampling_parameters, self.agent_id)
        self.action_history = []
        self.dialogue_history = []
        self.containers_name = []
        self.goal_objects_name = []
        self.rooms_name = []
        self.roomname2id = {}
        self.unsatisfied = {}
        self.steps = 0
        self.communication = True
        # self.location = None
        # self.last_location = None
        self.plan = None
        self.stuck = 0
        self.current_room = None
        self.last_room = None
        self.grabbed_objects = None
        self.opponent_grabbed_objects = []
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None
        self.id2node = {}
        self.id_inside_room = {}
        self.satisfied = []
        self.reachable_objects = []
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }


    @property
    def all_relative_name(self) -> list:
        return self.containers_name + self.goal_objects_name + self.rooms_name + ['character']
    
    def goexplore(self):
        target_room_id = int(self.plan.split(' ')[-1][1:-1])
        if self.current_room['id'] == target_room_id:
            self.plan = None
            return None
        return self.plan.replace('[goexplore]', '[walktowards]')
    
    
    def gocheck(self):
        assert len(self.grabbed_objects) < 2 # must have at least one free hands
        target_container_id = int(self.plan.split(' ')[-1][1:-1])
        target_container_name = self.plan.split(' ')[1]
        target_container_room = self.id_inside_room[target_container_id]
        if self.current_room['class_name'] != target_container_room:
            return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]
        if 'OPEN' in target_container['states']:
            self.plan = None
            return None
        if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.plan.replace('[gocheck]', '[open]') # conflict will work right?
        else:
            return self.plan.replace('[gocheck]', '[walktowards]')


    def gograb(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        target_object_name = self.plan.split(' ')[1]
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.plan = None
            return None
        assert len(self.grabbed_objects) < 2 # must have at least one free hands

        target_object_room = self.id_inside_room[target_object_id]
        if self.current_room['class_name'] != target_object_room:
            return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
            if self.debug:
                print(f"not here any more!")
            self.plan = None
            return None
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.plan.replace('[gograb]', '[grab]')
        else:
            return self.plan.replace('[gograb]', '[walktowards]')
    
    def goput(self):
        # if len(self.progress['goal_location_room']) > 1: # should be ruled out
        if len(self.grabbed_objects) == 0:
            self.plan = None
            return None
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        if self.current_room['class_name'] != target_room_name:
            return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
        if self.goal_location not in self.reachable_objects:
            return f"[walktowards] {self.goal_location}"
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]
        if "CONTAINERS" in y['properties']:
            if len(self.grabbed_objects) < 2 and'CLOSED' in y['states']:
                return self.plan.replace('[goput]', '[open]')
            else:
                action = '[putin]'
        else:
            action = '[putback]'
        x = self.id2node[self.grabbed_objects[0]]
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"


    def LLM_plan(self, get_text_obs=False, text_action=None):
        if get_text_obs:
            return self.LLM.get_text_obs(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id])
        else:
            if len(self.grabbed_objects) == 2:
                return f"[goput] {self.goal_location}", {}
            return self.LLM.run(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id], text_action=text_action)


    def check_progress(self, state, goal_spec):
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        for key, value in goal_spec.items():
            elements = key.split('_')
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and id2node[edge['from_id']]['class_name'] == elements[1]:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
                    # if self.debug:
                    #     print(satisfied)
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied


    def filter_graph(self, obs):
        relative_id = [node['id'] for node in obs['nodes'] if node['class_name'] in self.all_relative_name]
        relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]
        new_graph = {
            "edges": [edge for edge in obs['edges'] if
                      edge['from_id'] in relative_id and edge['to_id'] in relative_id],
            "nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
        }
    
        return new_graph
    
    def get_action(self, observation, goal, get_text_obs=False, get_anchor_obs=False, text_action=None,  cnt = -1):
        """
        :param observation: {"edges":[{'from_id', 'to_id', 'relation_type'}],
        "nodes":[{'id', 'category', 'class_name', 'prefab_name', 'obj_transform':{'position', 'rotation', 'scale'}, 'bounding_box':{'center','size'}, 'properties', 'states'}],
        "messages": [None, None]
        }
        :param goal:{predicate:[count, True, 2]}
        :return:
        """
        if self.communication:
            for i in range(len(observation["messages"])):
                if observation["messages"][i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i + 1]}: {observation['messages'][i]}")

        satisfied, unsatisfied = self.check_progress(observation, goal) 
        # print(f"satisfied: {satisfied}")
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied
            self.satisfied = satisfied
        obs = self.filter_graph(observation)
        self.grabbed_objects = []
        opponent_grabbed_objects = []
        self.reachable_objects = []
        self.id2node = {x['id']: x for x in obs['nodes']}
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id:
                if r == 'INSIDE':
                    self.current_room = self.id2node[y]
                elif r in ['HOLDS_RH', 'HOLDS_LH']:
                    self.grabbed_objects.append(y)
                elif r == 'CLOSE':
                    y = self.id2node[y]
                    self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
            elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
                opponent_grabbed_objects.append(self.id2node[y])

        unchecked_containers = []
        ungrabbed_objects = []
        for x in obs['nodes']:
            if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None: continue
                    j = None
                    for i, ungrab in enumerate(ungrabbed):
                        if x['id'] == ungrab['id']:
                            j = i
                    if j is not None:
                        ungrabbed.pop(j)
                continue
            self.id_inside_room[x['id']] = self.current_room['class_name']
            if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
                unchecked_containers.append(x)
            if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all([x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x['id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w in opponent_grabbed_objects]:
                ungrabbed_objects.append(x)

        if type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        info = {'graph': obs,
                "obs": {
                         "grabbed_objects": self.grabbed_objects,
                         "opponent_grabbed_objects": opponent_grabbed_objects,
                         "reachable_objects": self.reachable_objects,
                         "progress": {
                                "unchecked_containers": self.unchecked_containers,
                                "ungrabbed_objects": self.ungrabbed_objects,
                                      },
                        "satisfied": self.satisfied,
                        "current_room": self.current_room['class_name'],
                        },
                }
        if get_anchor_obs:
            obs_copy = dict(info["obs"])     # 创建浅拷贝
            obs_copy.pop("reachable_objects", None)   # 删除 progress，但不影响原 dict
            return obs_copy
        if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
            self.opponent_grabbed_objects = opponent_grabbed_objects
        action = None
        LM_times = 0
        if get_text_obs:
            text_obs = self.LLM_plan(get_text_obs)
            return text_obs
        while action is None:
            if text_action is not None and self.plan == None and cnt > 0:
                break
            if self.plan is None:
                if LM_times > 0:
                    print(info)
                if LM_times > 3:
                    print(text_action)                    
                    print(f"retrying LM_plan too many times")
                    plan = f"[wait]"
                    self.plan = plan
                    action = None
                    break
                    # raise Exception(f"retrying LM_plan too many times")
                if text_action is not None:
                    plan, a_info = self.LLM_plan(text_action=text_action)
                else:
                    plan, a_info = self.LLM_plan()
                if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"
                self.plan = plan
                self.action_history.append('[send_message]' if plan.startswith('[send_message]') else plan)
                a_info.update({"steps": self.steps})
                info.update({"LLM": a_info})
                LM_times += 1
            if self.plan.startswith('[goexplore]'):
                action = self.goexplore()
            elif self.plan.startswith('[gocheck]'):
                action = self.gocheck()
            elif self.plan.startswith('[gograb]'):
                action = self.gograb()
            elif self.plan.startswith('[goput]'):
                action = self.goput()
            elif self.plan.startswith('[send_message]'):
                action = self.plan[:]
                self.plan = None
            elif self.plan.startswith('[wait]'):
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        self.steps += 1
        info.update({"plan": self.plan,
                     })
        if action == self.last_action and self.current_room['class_name'] == self.last_room:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_action = action
        # self.last_location = self.location
        self.last_room = self.current_room
        if self.stuck > 20:
            print("Warning! stuck!")
            self.action_history[-1] += ' but unfinished'
            self.plan = None
            if type(self.id_inside_room[self.goal_location_id]) is list:
                target_room_name = self.id_inside_room[self.goal_location_id][0]
            else:
                target_room_name = self.id_inside_room[self.goal_location_id]
            action = f"[walktowards] {self.goal_location}"
            if self.current_room['class_name'] != target_room_name:
                action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            self.stuck = 0
    
        return action, info

    def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
        self.steps = 0
        self.containers_name = containers_name
        self.goal_objects_name = goal_objects_name
        self.rooms_name = rooms_name
        self.roomname2id = {x['class_name']: x['id'] for x in room_info}
        self.id2node = {x['id']: x for x in obs['nodes']}
        self.stuck = 0
        self.last_room = None
        self.unsatisfied = {k: v[0] for k, v in goal.items()}
        self.satisfied = []
        self.goal_location = list(goal.keys())[0].split('_')[-1]
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.id_inside_room = {self.goal_location_id: self.rooms_name[:], self.opponent_agent_id: None}
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.opponent_grabbed_objects = []
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id and r == 'INSIDE':
                self.current_room = self.id2node[y]
        self.plan = None
        self.action_history = [f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"]
        self.dialogue_history = []
        self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)
        

class CwahAgentMemoryWorker:
    """
    Ray remote actor that holds agents for one environment.
    Each worker manages a pair of agents (agent 1 and agent 2).
    """
    
    def __init__(self, env_id: int, base_url: str, lm_id: str, prompt_template: List[str], sampling_parameters: Dict[str, Any]):
        self.env_id = env_id
        self.agents = []
        self.base_url = base_url
        self.lm_id = lm_id
        self.prompt_template = prompt_template
        self.sampling_parameters = sampling_parameters
    
    def reset(self, obs, info, profile):
        """Reset agents for this environment"""
        self.agents = [
            (lambda a: (a.reset(
                obs[0],
                info["all_containers_name"],
                info["all_goal_objects_name"],
                info["all_room_name"],
                info["room_info"],
                info["goal_spec"][0]
            ), a)[1])(cwah_agent(agent_id=1, base_url=self.base_url, lm_id=self.lm_id,
                                prompt_template=self.prompt_template[0], 
                                sampling_parameters=self.sampling_parameters)),
            (lambda a: (a.reset(
                obs[1],
                info["all_containers_name"],
                info["all_goal_objects_name"],
                info["all_room_name"],
                info["room_info"],
                info["goal_spec"][1]
            ), a)[1])(cwah_agent(agent_id=2, base_url=self.base_url, lm_id=self.lm_id,
                                prompt_template=self.prompt_template[1].replace('$PLAYER_PROFILE$', profile), 
                                sampling_parameters=self.sampling_parameters)),
        ]
    
    def build_text_obs(self, obs, goal):
        """Build text observation for this environment"""
        return self.agents[0].get_action(obs[0], goal[0], get_text_obs=True)
    
    def build_anchor_obs(self, obs, goal):
        """Build anchor observations for both agents"""
        return {
            0: self.agents[0].get_action(obs[0], goal[0], get_anchor_obs=True),
            1: self.agents[1].get_action(obs[1], goal[1], get_anchor_obs=True)
        }
    
    def get_actions(self, action, obs, goal, action_info, cnt):
        """Get actions from both agents in this environment"""
        dict_action = {}
        dict_info = {}
        
        if len(action_info) == 0 or action_info[0]['plan'] is not None:
            for it, agent in enumerate(self.agents):
                if it == 0:
                    dict_action[it], dict_info[it] = agent.get_action(
                        observation=obs[it], 
                        goal=goal[it], 
                        text_action=action, 
                        cnt=cnt
                    )
                else:
                    dict_action[it], dict_info[it] = agent.get_action(
                        observation=obs[it], 
                        goal=goal[it]
                    )
        else:
            dict_info = action_info
        
        return dict_action, dict_info


class cwah_agent_memory_manager(BaseMemory):
    """
    Distributed memory manager using Ray for parallel agent processing.
    Each environment's agents run in a separate Ray actor.
    """
    
    def __init__(self, batch_size, resources_per_worker: Dict[str, float] = None):
        """
        Args:
            resources_per_worker: Ray resources allocation per worker
                                 e.g., {"num_cpus": 1, "num_gpus": 0.25}
        """
        self.workers = []
        self.batch_size = 0
        self.resources_per_worker = resources_per_worker or {"num_cpus": 0.25}
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
    
    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.workers[idx]
    
    def reset(self, batch_size: int, obs, infos: List[Dict[str, Any]], 
              base_url: str, lm_id: str = "gpt-4", 
              prompt_template: List[str] = None, 
              profiles: List[str] = None, 
              sampling_parameters: Dict[str, Any] = None):
        """
        Reset all workers with new environments.
        
        Args:
            batch_size: Number of environments
            obs: Observations for each environment
            infos: Info dicts for each environment
            base_url: LLM API base URL
            lm_id: Language model ID
            prompt_template: Prompt templates for agents
            profiles: Agent profiles for each environment
            sampling_parameters: LLM sampling parameters
        """
        # Clean up old workers
        if self.workers:
            for worker in self.workers:
                ray.kill(worker)
            self.workers = []
        
        # Create Ray remote class with resources
        AgentWorker = ray.remote(**self.resources_per_worker)(CwahAgentMemoryWorker)
        
        # Create workers for each environment
        self.workers = [
            AgentWorker.remote(
                env_id=i,
                base_url=base_url,
                lm_id=lm_id,
                prompt_template=prompt_template,
                sampling_parameters=sampling_parameters
            )
            for i in range(batch_size)
        ]
        
        # Reset all workers in parallel
        reset_futures = [
            self.workers[i].reset.remote(obs[i], infos[i], profiles[i])
            for i in range(batch_size)
        ]
        ray.get(reset_futures)  # Wait for all resets to complete
        
        self.batch_size = batch_size
        self.keys = None
    
    def build_text_obs(self, obs, goal):
        """Build text observations for all environments in parallel"""
        assert len(obs) == self.batch_size, \
            f"Number of observations ({len(obs)}) must equal batch_size ({self.batch_size})"
        
        # Send requests to all workers
        futures = [
            self.workers[i].build_text_obs.remote(obs[i], goal[i])
            for i in range(self.batch_size)
        ]
        
        # Collect results
        postprocess_text_obs = ray.get(futures)
        return postprocess_text_obs
    
    def build_anchor_obs(self, obs, goal):
        """Build anchor observations for all environments in parallel"""
        assert len(obs) == self.batch_size, \
            f"Number of observations ({len(obs)}) must equal batch_size ({self.batch_size})"
        
        # Send requests to all workers
        futures = [
            self.workers[i].build_anchor_obs.remote(obs[i], goal[i])
            for i in range(self.batch_size)
        ]
        
        # Collect results
        postprocess_anchor_obs = ray.get(futures)
        return postprocess_anchor_obs
    
    def get_actions(self, actions, obs, goal, action_infos, cnt):
        """Get actions from all environments in parallel"""
        assert len(actions) == self.batch_size, \
            f"Number of actions ({len(actions)}) must equal batch_size ({self.batch_size})"
        
        # Send action requests to all workers
        futures = [
            self.workers[i].get_actions.remote(
                actions[i],
                obs[i],
                goal[i],
                action_infos[i] if action_infos else [],
                cnt
            )
            for i in range(self.batch_size)
        ]
        
        # Collect results
        results = ray.get(futures)
        
        dict_action_list = [result[0] for result in results]
        dict_info_list = [result[1] for result in results]
        
        return dict_action_list, dict_info_list
    
    def store(self, record: Dict[str, List[Any]]):
        """Stores a new batch of records into memory."""
        # TODO: Implement if needed for your use case
        pass
    
    def fetch(self, step: int):
        """Fetches memory records at a specific time step across all environments."""
        # TODO: Implement if needed for your use case
        pass
    
    def close(self):
        """Close all workers"""
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []

# class cwah_agent_memory(BaseMemory):
#     """
#     Memory manager: responsible for storing & fetching per‑environment history records.
#     """
#     def __init__(self):
#         self._data = None
#         self.keys = None
#         self.batch_size = 0

#     def __len__(self):
#         return len(self._data)

#     def __getitem__(self, idx):
#         return self._data[idx]

#     def reset(self, batch_size: int, obs, infos: List[Dict[str, Any]], base_url: str, lm_id: str = "gpt-4", prompt_template: str = None, profiles: str = None, sampling_parameters: Dict[str, Any] = None):
#         if self._data is not None:
#             self._data.clear()
#         self._data = [
#             [
#                 (lambda a: (a.reset(
#                     obs[i][0],
#                     infos[i]["all_containers_name"],
#                     infos[i]["all_goal_objects_name"],
#                     infos[i]["all_room_name"],
#                     infos[i]["room_info"],
#                     infos[i]["goal_spec"][0]
#                 ), a)[1])(cwah_agent(agent_id=1, base_url=base_url, lm_id=lm_id,
#                                     prompt_template=prompt_template[0], sampling_parameters=sampling_parameters)),
#                 (lambda a: (a.reset(
#                     obs[i][1],
#                     infos[i]["all_containers_name"],
#                     infos[i]["all_goal_objects_name"],
#                     infos[i]["all_room_name"],
#                     infos[i]["room_info"],
#                     infos[i]["goal_spec"][1]
#                 ), a)[1])(cwah_agent(agent_id=2, base_url=base_url, lm_id=lm_id,
#                                     prompt_template=prompt_template[1].replace('$PLAYER_PROFILE$', profiles[i]), sampling_parameters=sampling_parameters)),
#             ]
#             for i in range(batch_size)
#         ]
#         self.batch_size = batch_size
#         self.keys = None
    
#     def build_text_obs(self, obs, goal):
#         postprocess_text_obs = []
#         for i, agents in enumerate(self._data):
#             postprocess_text_obs.append(agents[0].get_action(obs[i][0], goal[i][0], get_text_obs=True))
#         return postprocess_text_obs

#     def build_anchor_obs(self, obs, goal):
#         postprocess_anchor_obs = []
#         for i, agents in enumerate(self._data):
#             postprocess_anchor_obs.append({0: agents[0].get_action(obs[i][0], goal[i][0], get_anchor_obs=True), 1: agents[1].get_action(obs[i][1], goal[i][1], get_anchor_obs=True)})
#         return postprocess_anchor_obs
    
#     def get_actions(self, actions, obs, goal, action_infos, cnt):
#         dict_action_list = []
#         dict_info_list = []
#         for i, agents in enumerate(self._data):
#             dict_action = {}
#             dict_info = {}
#             if len(action_infos) == 0 or action_infos[i][0]['plan'] is not None:
#                 for it, agent in enumerate(agents):
#                     if it == 0:
#                         dict_action[it], dict_info[it] = agent.get_action(observation=obs[i][it], goal=goal[i][it], text_action=actions[i], cnt = cnt)
#                     else:
#                         dict_action[it], dict_info[it] = agent.get_action(observation=obs[i][it], goal=goal[i][it])
#             else:
#                 dict_info = action_infos[i]
#             dict_action_list.append(dict_action)
#             dict_info_list.append(dict_info)
#         return dict_action_list, dict_info_list
    
#     def store(self, record: Dict[str, List[Any]]):
#         """
#         Stores a new batch of records into memory.
#         """
#         pass

#     def fetch(self, step: int):
#         """
#         Fetches memory records at a specific time step across all environments.
#         """
#         pass