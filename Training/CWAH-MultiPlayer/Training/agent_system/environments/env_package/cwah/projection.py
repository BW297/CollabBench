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
import re

def judge_action(available_actions, text):

    if "[send_message]" in text:
        return 1
        
    for i in range(len(available_actions)):
        action = available_actions[i]
        if action in text:
            return 1

    for i in range(len(available_actions)):
        action = available_actions[i]
        option = chr(ord('A') + i)
        # txt = text.lower()
        if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
            return 1
    # print("WARNING! Fuzzy match!")
    for i in range(len(available_actions)):
        action = available_actions[i]
        if i == 0:
            continue
        act, name, id = action.split(' ')
        option = chr(ord('A') + i)
        if f"{option} " in text or act in text or name in text or id in text:
            return 1
    # print(text)
    # print("WARNING! No available action parsed!!! Random choose one")
    return 0

def cwah_projection(actions: List[str], memory, obs, goal, action_infos, cnt, available_actions):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        if start_idx == -1 or end_idx == -1:
            # If we can't find a valid <action>...</action> block, mark as invalid
            # actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
            continue

        # Extract just the content between the tags
        extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
        # if "[send_message]" in extracted_action:
        #     pattern = r'^\[send_message\] <.*>$'
        #     if not re.match(pattern, extracted_action):
        #         continue
        
        if judge_action(available_actions[i], extracted_action) == 0:
            continue

        valids[i] = 1
        # check <think>...</think>
        # think_start_idx = original_str.find("<think>")
        # think_end_idx = original_str.find("</think>")
        think_start_idx = original_str.find("<cot>")
        think_end_idx = original_str.find("</cot>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check <message>...</message>
        message_start_idx = original_str.find("<message>")
        message_end_idx = original_str.find("</message>")
        if message_start_idx == -1 or message_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    actions, infos = memory.get_actions(actions, obs, goal, action_infos, cnt)

    return actions, valids, infos