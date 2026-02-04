import itertools, os, json, re
from collections import defaultdict
import numpy as np
import pandas as pd
import pkg_resources
import sys 
import copy 
from .modules import Module
import sys
import json
from string import Template
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
env_action = {
    "cramped_room": {
        "P0": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "set_message"
        ],
        "P1": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "set_message"
        ]
    },

    "asymmetric_advantages": {
        "P0": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "wait(x)", "set_message"
        ],
        "P1": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "wait(x)", "set_message"
        ]
    },

    "coordination_ring": {
        "P0": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "wait(x)", "set_message"
        ],
        "P1": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "wait(x)", "set_message"
        ]
    },

    "counter_circuit": {
        "P0": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "set_message"
        ],
        "P1": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "set_message"
        ]
    },

    "forced_coordination": {
        "P0": [
            "pickup_onion", "put_onion_in_pot", "pickup_dish",
            "fill_dish_with_soup", "deliver_soup",
            "place_obj_on_counter", "wait(x)", "set_message"
        ],
        "P1": [
            "pickup_onion", "pickup_dish",
            "place_obj_on_counter", "wait(x)", "set_message"
        ]
    }
}
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import find_path 
from overcooked_ai_py.planning.search import get_intersect_counter 
from overcooked_ai_py.planning.search import query_counter_states 
np.random.seed(2)
cwd = os.getcwd()
PROMPT_DIR = os.path.join(cwd, "prompts")

NAME_TO_ACTION = {
	"NORTH": Direction.NORTH,
	"SOUTH": Direction.SOUTH,
	"EAST": Direction.EAST,
	"WEST": Direction.WEST,
	"INTERACT": Action.INTERACT,
	"STAY": Action.STAY
}


class ProAgent(object):
	"""
	This agent uses GPT-3.5 to generate actions.
	"""
	def __init__(self, model="gpt-3.5-turbo-0301"):
		self.agent_index = None
		self.model = model
		self.behavior_description = ""


	def set_agent_index(self, agent_index):
		raise NotImplementedError

	def action(self, state):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError


class ProMediumLevelAgent(ProAgent):
	"""
	This agent default to use GPT-3.5 to generate medium level actions.
	"""
	def __init__(
			self,
			mlam,
			layout,
			model='gpt-3.5-turbo-0301',
			prompt_level='l2-ap', # ['l1-p', 'l2-ap', 'l3-aip']
			belief_revision=False,
			retrival_method="recent_k",
			K=1, 
			auto_unstuck=False,
			controller_mode='new', # the default overcooked-ai Greedy controller
			debug_mode='N', 
			agent_index=None,
			outdir = None,
			using_big_5=True,
			big_5=None,
			level=None,
			agent_url=None,
			agent_model=None,
			agent_api_key=None,
			player_url=None,
			player_model=None,
			player_api_key=None
	):
		super().__init__(model=model)

		self.trace = True 
		self.debug_mode = 'Y' 
		self.controller_mode = controller_mode 
		self.mlam = mlam
		self.layout = layout
		self.mdp = self.mlam.mdp
		
		self.out_dir = outdir 
		self.agent_index = agent_index

		self.prompt_level = prompt_level
		self.belief_revision = belief_revision

		self.retrival_method = retrival_method
		self.K = K
		
		self.prev_state = None
		self.auto_unstuck = auto_unstuck

		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.pot_id_to_pos = []

		self.using_big_5 = using_big_5
		self.big_5 = big_5
		self.level = level
		self.trace_list = []
		
		# Store API configuration for agent and player
		self.agent_url = agent_url
		self.agent_model = agent_model
		self.agent_api_key = agent_api_key
		self.player_url = player_url
		self.player_model = player_model
		self.player_api_key = player_api_key
		if self.using_big_5:
			big_5_file = os.path.join(PROMPT_DIR, 'cook.csv')
			df = pd.read_csv(big_5_file)
			self.big_5_desp = {}
			for _, row in df.iterrows():
				value_dict = {}
				value_dict['Profile'] = row['Profile']
				key = f"{row['Task']}_{row['Cluster']}"
				self.big_5_desp[key] = value_dict

		self.layout_prompt = self.generate_layout_prompt()
		with open("actions.json", "r") as file:
				self.legal_actions = json.load(file)


	def set_mdp(self, mdp):
		self.mdp = mdp

	def create_gptmodule(self, module_name, file_type='txt', retrival_method='recent_k', K=10):
		print(f"\n--->Initializing GPT {module_name}<---\n")    

		# prompt_file = os.path.join(PROMPT_DIR, self.model, module_name, self.layout+f'_{self.agent_index}.'+file_type)

		if "gpt" in self.model or "text-davinci" in self.model:
			model_name = "gpt"
		elif "claude" in self.model:
			model_name = "claude"
		else:
			model_name= "gpt"
	
		if module_name == "planner":
			prompt_file = os.path.join(PROMPT_DIR, f'refine_{self.agent_index}.{file_type}')
		elif module_name == "explainer":
			prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, f'player{self.agent_index}.{file_type}')

		else:
			raise Exception(f"Module {module_name} not supported.")


		with open(prompt_file, "r", encoding="utf-8") as f:
			prompt_content = f.read()
			if file_type == 'json':
				messages = json.loads(prompt_content)
			elif file_type == 'txt':
				if self.using_big_5 and module_name in ["planner"]:
					string_tem = Template(prompt_content)

					try:
						key = f"{self.layout}_{str(self.level)}"
						if self.agent_index==1 and self.using_big_5:
							personality_profile = self.big_5_desp[key]['Profile']
							self.behavior_description=personality_profile
						else:
							personality_profile='Unknown'
							self.behavior_description=personality_profile
					except KeyError:
						print(f"Warning: Personality level {self.level} not found for {self.layout}. Falling back to cluster 0.")
						personality_profile = "Unknown"
						self.behavior_description=personality_profile

					message = string_tem.safe_substitute(personality_def=personality_profile)

					if "Task Instructions:" in message:
						system_content = message.split("Task Instructions:")[0].strip()
						messages = [{"role": "system", "content": system_content}]
					else:
						messages = [{"role": "system", "content": prompt_content}]

				else:
					if "Task Instructions:" in prompt_content:
						system_content = prompt_content.split("Task Instructions:")[0].strip()
						messages = [{"role": "system", "content": system_content}]
					else:
						messages = [{"role": "system", "content": prompt_content}]

			else:
				print("Unsupported file format.")
		
		module = Module(
			messages, 
			self.model, 
			retrival_method, 
			K, 
			self.agent_index,
			agent_url=self.agent_url,
			agent_model=self.agent_model,
			agent_api_key=self.agent_api_key,
			player_url=self.player_url,
			player_model=self.player_model,
			player_api_key=self.player_api_key
		)
		if module_name == "planner" and file_type == 'txt':

			dynamic_vars = ['$step', '$scene', '$chat']
			module.uses_new_prompt_format = any(var in prompt_content for var in dynamic_vars)
			if module.uses_new_prompt_format:

				module.prompt_template = prompt_content
		else:
			module.uses_new_prompt_format = False
		
		return module

	def reset(self):
		self.planner.reset()
		self.explainer.reset()
		self.prev_state = None
		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.current_timestep = 0
		self.teammate_ml_actions_dict = {}
		self.teammate_intentions_dict = {}


	def set_agent_index(self, agent_index):
		self.agent_index = agent_index
		self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K)
		self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=self.K)

		print(self.planner.instruction_head_list[0]['content'])
	
	
	def generate_conversation_list(self, chat_list, K=5):
		"""
		Generate conversation prompt from chat_list, displaying dialogue between two players.
		
		chat_list format example:
		[
			{"ID": 0, "message": "Hello!"},
			{"ID": 1, "message": "Hi, how are you?"},
			{"ID": 0, "message": "I'm good, thanks!"}
		]
		
		K: Display the most recent K messages
		"""
		if not chat_list:
			return "No previous chat messages. "

		# Keep only the most recent K messages
		recent_chats = chat_list[-K:]

		# Build conversation string
		conversation_lines = []
		for chat in recent_chats:
			speaker = f"Player {chat['ID']}"
			message = chat['message']
			conversation_lines.append(f"{speaker}: {message}")

		return conversation_lines

	def generate_layout_prompt(self):
		layout_prompt_dict = {
			"onion_dispenser": " <Onion Dispenser {id}>",
			"dish_dispenser": " <Dish Dispenser {id}>",
			"tomato_dispenser": " <Tomato Dispenser {id}>",
			"serving": " <Serving Loc {id}>",
			"pot": " <Pot {id}>",
		}
		layout_prompt = "Here's the layout of the kitchen:"
		for obj_type, prompt_template in layout_prompt_dict.items():
			locations = getattr(self.mdp, f"get_{obj_type}_locations")()
			for obj_id, obj_pos in enumerate(locations):
				layout_prompt += prompt_template.format(id=obj_id) + ","
				if obj_type == "pot":
					self.pot_id_to_pos.append(obj_pos)
		layout_prompt = layout_prompt[:-1] + ". "
		return layout_prompt
	  
	def generate_state_prompt(self, state):
		ego = state.players[self.agent_index]
		teammate = state.players[1 - self.agent_index]


		time_prompt = f"Current scene: "
  
		ego_object = ego.held_object.name if ego.held_object else "nothing"
		teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
		ego_state_prompt = f"<Player {self.agent_index}> holds "
		if ego_object == 'soup':
			ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
		elif ego_object == 'nothing':
			ego_state_prompt += f"{ego_object}. "
		else:
			ego_state_prompt += f"one {ego_object}. "
		
		teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
		if teammate_object == 'soup':
			teammate_state_prompt += f"a dish with {teammate_object}. "
		elif teammate_object == "nothing":
			teammate_state_prompt += f"{teammate_object}. "
		else:
			teammate_state_prompt += f"one {teammate_object}. "

		
		kitchen_state_prompt = "Kitchen states: "
		prompt_dict = {
			"empty": "<Pot {id}> is empty; ",
			"cooking": "<Pot {id}> starts cooking, the soup will be ready after {t} timesteps; ",
			"ready": "<Pot {id}> has already cooked the soup; ",
			"1_items": "<Pot {id}> has 1 onion; ",
			"2_items": "<Pot {id}> has 2 onions; ",
			"3_items": "<Pot {id}> has 3 onions and is full; "
		}

		pot_states_dict = self.mdp.get_pot_states(state)   

		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			for key in pot_states_dict.keys():
				if key == "cooking":
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						soup_object = state.get_object(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
				else:
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id) 
		
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			for key in pot_states_dict.keys():
				if key == "empty":
					for pos in pot_states_dict[key]: 
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id)     
				else: # key = 'onion' or 'tomota'
					for soup_key in pot_states_dict[key].keys():
						# soup_key: ready, cooking, partially_full
						for pos in pot_states_dict[key][soup_key]:
							pot_id = self.pot_id_to_pos.index(pos)
							soup_object = state.get_object(pos)
							soup_type, num_items, cook_time = soup_object.state
							if soup_key == "cooking":
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
							elif soup_key == "partially_full":
								pass
							else:
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)
 

		intersect_counters = get_intersect_counter(
								state.players_pos_and_or[self.agent_index], 
								state.players_pos_and_or[1 - self.agent_index], 
								self.mdp, 
								self.mlam
							)
		counter_states = query_counter_states(self.mdp, state)  

		if self.layout == 'forced_coordination': 
			kitchen_state_prompt += '{} counters can be visited by <Player {}>. Their states are as follows: '.format(len(intersect_counters), self.agent_index)
			count_states = {}  
			for i in intersect_counters:  
				obj_i = 'nothing' 
				if counter_states[i] != ' ': 
					obj_i = counter_states[i]                
				if obj_i in count_states:  
					count_states[obj_i] += 1
				else: 
					count_states[obj_i]  = 1 
			total_obj = ['onion', 'dish']
			for i in count_states:   
				if i == 'nothing': 
					continue 
				kitchen_state_prompt += f'{count_states[i]} counters have {i}. '   
			for i in total_obj: 
				if i not in count_states:        
					kitchen_state_prompt += f'No counters have {i}. ' 

		if self.layout == "forced_coordination":
			teammate_state_prompt=""
		return (self.layout_prompt + time_prompt + ego_state_prompt +
				teammate_state_prompt + kitchen_state_prompt)

	def generate_belief_prompt(self):
		ego_id = self.agent_index
		intention_prompt = f"All <Player {ego_id}> infered intentions about <Player {1-ego_id}>: {self.teammate_intentions_dict}.\n"
		real_behavior_prompt = f"<Player {1-ego_id}> real behaviors: {self.teammate_ml_actions_dict}.\n"
		belief_prompt = intention_prompt + real_behavior_prompt
		return belief_prompt
	
	##################
	'''
	The followings are the Planner part
	'''
	##################
	def generate_conversation_prompt(self, chat_list, K=5):
		"""
		Generate conversation prompt from chat_list, displaying dialogue between two players.
		
		chat_list format example:
		[
			{"ID": 0, "message": "Hello!"},
			{"ID": 1, "message": "Hi, how are you?"},
			{"ID": 0, "message": "I'm good, thanks!"}
		]
		
		K: Display the most recent K messages
		"""
		if not chat_list:
			return "No previous chat messages. "

		# Keep only the most recent K messages
		recent_chats = chat_list[-K:]

		# Build conversation string
		conversation_lines = []
		for chat in recent_chats:
			speaker = f"Player {chat['ID']}"
			message = chat['message']
			conversation_lines.append(f"{speaker}: {message}")

		# Concatenate into final prompt
		prompt = "\n".join(conversation_lines)
		return prompt
     

	def action(self, state):

		start_pos_and_or = state.players_pos_and_or[self.agent_index]

		# only use to record the teammate ml_action, 
		# if teammate finish ml_action in t-1, it will record in s_t, 
		# otherwise, s_t will just record None,
		# and we here check this information and store it into proagent
		self.current_timestep = state.timestep
		
		# if current ml action does not exist, generate a new one
		if self.current_ml_action is None:
			self.current_ml_action = self.generate_ml_action(state)

		# if the current ml action is in process, Player{self.agent_index} done, else generate a new one
		if self.current_ml_action_steps > 0:
			current_ml_action_done = self.check_current_ml_action_done(state)
			if current_ml_action_done:
				# generate a new ml action
				self.generate_success_feedback(state)
				self.current_ml_action = self.generate_ml_action(state)

		# Validate the generated action (should be legal since we only generate from legal actions)
		# But keep a simple check as a safety measure
		if not self.validate_current_ml_action(state):
			# This should rarely happen since we only generate from legal actions
			# But if it does, fallback to wait(1)
			print(f'Warning: P{self.agent_index} generated action {self.current_ml_action} is invalid, falling back to wait(1)')
		
			self.current_ml_action = "wait(1)"
			self.time_to_wait = 1
		
		self.trace = True 
		if "wait" in self.current_ml_action:
			self.current_ml_action_steps += 1
			self.time_to_wait -= 1
			lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
			chosen_action =lis_actions[np.random.randint(0,len(lis_actions))]

			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				self.prev_state = state
				return chosen_action, {}
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				self.prev_state = state
				return chosen_action
		elif "message" in self.current_ml_action:
			state.steps[str(self.agent_index)]=0
			self.current_ml_action_steps += 1
			lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
			chosen_action = lis_actions[np.random.randint(0, len(lis_actions))]

			
			# Check if message has already been extracted from the response
			extracted_message = None
			if self.agent_index == 0 and state.behavior_list.get("P0") and len(state.behavior_list["P0"]) > 0:
				extracted_message = state.behavior_list["P0"][-1].get("ExtractedMessage") or state.behavior_list["P0"][-1].get("Message")
			elif self.agent_index == 1 and state.behavior_list.get("P1") and len(state.behavior_list["P1"]) > 0:
				extracted_message = state.behavior_list["P1"][-1].get("ExtractedMessage") or state.behavior_list["P1"][-1].get("Message")
			
			if extracted_message:
				# Use the extracted message
				response = extracted_message
			else:
				response = None
			
			if self.agent_index == 0:
				state.chat_list["P0"].append({"scene": self.generate_state_prompt(state),"content": response})
				
			else:
				state.chat_list["P1"].append({"scene": self.generate_state_prompt(state),"content": response})
			state.chat_all.append({"ID":self.agent_index, "message":response})
			if self.agent_index == 0:
				assert state.behavior_list["P0"][-1]["Plan"] == "set_message"
				state.behavior_list["P0"][-1]["Plan"]+=f": {response}"
			else:
				assert state.behavior_list["P1"][-1]["Plan"] == "set_message"
				state.behavior_list["P1"][-1]["Plan"]+=f": {response}"

			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				self.prev_state = state
				return chosen_action, {}
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				self.prev_state = state
				return chosen_action
		else:
			possible_motion_goals = self.find_motion_goals(state)    
			current_motion_goal, chosen_action = self.choose_motion_goal(
				start_pos_and_or, 
				possible_motion_goals, 
				state
			)
		# if "wait" in self.current_ml_action: 
		# 	print(f'current motion goal for P{self.agent_index} is wait') 
		# else: 
		# 	if current_motion_goal is None: 
		# 		current_motion_goal = 'None' 
		# 	print(f'current motion goal for P{self.agent_index} is {current_motion_goal}') 


		if self.auto_unstuck and chosen_action != Action.INTERACT:
			if (
					self.prev_state is not None
					and state.players
					== self.prev_state.players
			):
				if self.agent_index == 0:
					joint_actions = list(
						itertools.product(Action.ALL_ACTIONS, [Action.STAY])
					)
				elif self.agent_index == 1:
					joint_actions = list(
						itertools.product([Action.STAY], Action.ALL_ACTIONS)
					)
				else:
					raise ValueError("Player index not recognized")

				unblocking_joint_actions = []
				for j_a in joint_actions:
					if j_a != [Action.INTERACT,Action.STAY] and  j_a != [Action.STAY,Action.INTERACT]:
						if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
							new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
						elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
							new_state, _, _ = self.mlam.mdp.get_state_transition(state, j_a)		
						if (
								new_state.players_pos_and_or
								!= self.prev_state.players_pos_and_or
							):
							unblocking_joint_actions.append(j_a)
				unblocking_joint_actions.append([Action.STAY, Action.STAY])
				chosen_action = unblocking_joint_actions[
					np.random.choice(len(unblocking_joint_actions))
				][self.agent_index]

		self.prev_state = state
		if chosen_action is None:
			if self.agent_index == 0:
				state.behavior_list["P0"].pop()
				state.state_back["P0"].pop()
			else:
				state.behavior_list["P1"].pop()
				state.state_back["P1"].pop()
				state.state_back["P1_done"].pop()
			self.current_ml_action = "wait(1)"
			result={"Analysis": "Re-generating ml_action failed for over 3 times, so wait(1) is given.","Plan":self.current_ml_action}
			P0_state_back={"state":self.generate_state_prompt(state),"action":self.current_ml_action}
			if len(state.behavior_list["P0"])==0:
				P1_done_human_act="None"
				P1_human_act=""
			elif len(state.behavior_list["P0"])==1:
				P1_done_human_act="None"
				P1_human_act=state.behavior_list["P0"][-1]['Plan']
			else:
				P1_done_human_act=state.behavior_list["P0"][-2]['Plan']
				P1_human_act=state.behavior_list["P0"][-1]['Plan']
			if len(state.behavior_list["P1"])==0:
				Agent_act="None"
			else:
				Agent_act=state.behavior_list["P1"][-1]['Plan']
			P1_done_state_back={"state":self.behavior_description,"Human_pre_action":P1_done_human_act,"Agent_pre_action":Agent_act,"action":self.current_ml_action}
			P1_state_back={"state":self.behavior_description,"Human_pre_action":P1_human_act,"Agent_pre_action":Agent_act,"action":self.current_ml_action}
			if self.agent_index == 0:
				state.behavior_list["P0"].append(result)
				state.state_back["P0"].append(P0_state_back)
			else:
				state.behavior_list["P1"].append(result)
				state.state_back["P1"].append(P1_state_back)
				state.state_back["P1_done"].append(P1_done_state_back)
			self.time_to_wait = 1
			chosen_action = Action.STAY
		self.current_ml_action_steps += 1

		# print(f'ml_action = {self.current_ml_action}') 
		# print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')
		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			return chosen_action, {}
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			return chosen_action
	


	def parse_xml_response(self, response):
		result = {"Analysis": "", "Message": "", "Plan": ""}

		reasoning_pattern = r'<think>(.*?)</think>'
		reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)
		if reasoning_match:
			result["Analysis"] = reasoning_match.group(1).strip()
		else:
			result["Analysis"] =response

		message_pattern = r'<message>(.*?)</message>'
		message_match = re.search(message_pattern, response, re.DOTALL)
		if message_match:
			result["Message"] = message_match.group(1).strip() if message_match else response
		
		action_pattern = r'<action>(.*?)</action>'
		action_match = re.search(action_pattern, response, re.DOTALL)
		if action_match:
			result["Plan"] = action_match.group(1).strip() if action_match else response
		
		return result

	def parse_ml_action(self, action_string): 

		try: 
			ml_action = action_string.split()[0]
		except: 
			print('failed on 528') 
			action_string = 'wait(1)'
			ml_action = action_string
			# ml_action = 'wait(1)' 

		if "place" in action_string:
			ml_action = "place_obj_on_counter"
		elif "pick" in action_string:
			if "onion" in action_string:
				ml_action = "pickup_onion"
			elif "tomato" in action_string:
				ml_action = "pickup_tomato"
			elif "dish" in action_string:
				ml_action = "pickup_dish"
		elif "put" in action_string:
			if "onion" in action_string:
				ml_action = "put_onion_in_pot"
			elif "tomato" in action_string:
				ml_action = "put_tomato_in_pot"
		elif "fill" in action_string:   
			ml_action = "fill_dish_with_soup"
		elif "deliver" in action_string:
			ml_action = "deliver_soup"
		elif "message" in action_string:
			ml_action = "set_message"
		elif "wait" not in action_string:
			ml_action='wait(1)'  
			action_string = ml_action
		if "wait" in action_string:
			
			def parse_wait_string(s):
				# Check if it's just "wait"
				if s == "wait":
					return 1

				# Remove 'wait' and other characters from the string
				s = s.replace('wait', '').replace('(', '').replace(')', '').replace('"', '').replace('.', '') 
				# If it's a number, return it as an integer
				if s.isdigit():
					return int(s)
				# If it's not a number, return a default value or raise an exception
				return 1
			if self.layout == 'forced_coordination': 
				self.time_to_wait = max(3, parse_wait_string(action_string))
			else: 
				self.time_to_wait = parse_wait_string(action_string)    
			
			ml_action = f"wait({self.time_to_wait})"

		else:
			pass
		
		# aviod to generate two skill, eg, Plan for Player 0: "deliver_soup(), pickup(onion)".
		if "," in ml_action:
			ml_action = ml_action.split(',')[0].strip()

		            
		return ml_action    


	def generate_legal_actions_prompt_section(self, state):
		"""
		Generate the "Available actions" section of the prompt with only legal actions.
		Returns a formatted string.
		"""
		legal_actions = self.legal_actions[self.layout][str(self.agent_index)]
		
		if not legal_actions:
			return "    - No legal actions available at this moment."
		
		action_lines = []
		available_action_list=[]
		for item in legal_actions:
			action = item['action']
			if self.is_ml_action_legal(state, action):
				conditions = item['condition']
				
				# Format action name to match prompt format
				if "pickup" in action and "onion" in action:
					action_display = "pickup(onion)"

				elif "pickup" in action and "dish" in action:
					action_display = "pickup(dish)"
				elif action == "wait(x)":
					action_display = "wait(x)"
				elif action.startswith("wait("):
					action_display = action
				elif action == "set_message":
					action_display = "set_message()"
				elif action.endswith("()"):
					action_display = action
				else:
					# For actions like "put_onion_in_pot", add () if not present
					if "(" not in action:
						action_display = action + "()"
					else:
						action_display = action
				
				# Build action line with conditions
				action_line = f"    - {action_display}"
				if conditions:
					for condition in conditions:
						action_line += f"\n        - {condition}"
				action_lines.append(action_line)
		# available_action_list=[]
		# for action in legal_actions:
				available_action_list.append(action)
		
		return "\n".join(action_lines), ",".join(available_action_list)


	def generate_ml_action(self, state):
		"""
		Selects a medium level action for the current state.
		Motion goals can be thought of instructions of the form:
			[do X] at location [Y]

		In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
		a simple set of  heuristics based on the current state.

		Effectively, will return a list of all possible locations Y in which the selected
		medium level action X can be performed.
		"""


		# Use new format: build user message containing dynamic variables
		if hasattr(self.planner, 'prompt_template'):
			prompt_template = self.planner.prompt_template

		legal_actions_section, legal_actions_list = self.generate_legal_actions_prompt_section(state)
		
		task_start = prompt_template.find("Task Instructions:")
		if task_start != -1:
			task_section = prompt_template[task_start:]
		else:
			task_section = prompt_template
		
		# Replace dynamic variables
		from string import Template
		# Handle cases like $ action (with spaces), fix template first
		template = Template(task_section)
		try:
			user_content = template.substitute(
				actions_des=legal_actions_section,
				actions_list=legal_actions_list,
				step=state.timestep,
				scene=self.generate_state_prompt(state),
				chat=self.generate_conversation_prompt(chat_list=state.chat_all, K=3),
				
			)
		except KeyError as e:
			# If a variable doesn't exist, use safe_substitute
			safe_dict = {
				'actions_des': legal_actions_section,
				'actions_list': legal_actions_list,
				'step': state.timestep,
				'scene': self.generate_state_prompt(state),
				'chat': self.generate_conversation_prompt(chat_list=state.chat_all, K=3),
			
			}
			user_content = template.safe_substitute(**safe_dict)
		
		state_prompt = user_content
		stop_keywords = None  # New format doesn't need stop condition


		print(f"\n\n### Observation module to GPT\n")   
		print(f"{state_prompt}")

		state_message = {"role": "user", "content": state_prompt}
		self.planner.current_user_message = state_message
		response = self.planner.query(key=self.openai_api_key(), stop=stop_keywords, trace = self.trace)
		
		if 'wait' not in response:
			self.planner.add_msg_to_dialog_history(state_message) 
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
		
		print(f"\n\n\n### GPT Planner module\n")   
		print("====== GPT Query ======")
		print(response)  
		
		result = self.parse_xml_response(response)
		ml_action = self.parse_ml_action(result['Plan'])
		result["Plan"] = ml_action

		if "Message" not in result:
			result["Message"] = ""
		
		ml_action = result["Plan"]  # Use the final Plan as ml_action
		
		conversation_history = self.generate_conversation_list(chat_list=state.chat_all, K=3)
		action_history = state.behavior_list.get(f"P{self.agent_index}", []).copy()
		
		P0_state_back={"state":self.generate_state_prompt(state),"action":ml_action}
		if len(state.behavior_list["P0"])==0:
			P1_done_human_act="None"
			P1_human_act=""
		elif len(state.behavior_list["P0"])==1:
			P1_done_human_act="None"
			P1_human_act=state.behavior_list["P0"][-1]['Plan']
		else:
			P1_done_human_act=state.behavior_list["P0"][-2]['Plan']
			P1_human_act=state.behavior_list["P0"][-1]['Plan']
		if len(state.behavior_list["P1"])==0:
			Agent_act="None"
		else:
			Agent_act=state.behavior_list["P1"][-1]['Plan']
		P1_done_state_back={"state":self.behavior_description,"Human_pre_action":P1_done_human_act,"Agent_pre_action":Agent_act,"action":ml_action}
		P1_state_back={"state":self.behavior_description,"Human_pre_action":P1_human_act,"Agent_pre_action":Agent_act,"action":ml_action}
		if self.agent_index == 0:
			state.behavior_list["P0"].append(result)
			state.scene_list["P0"].append(state_prompt)
			state.state_back["P0"].append(P0_state_back)

		else:
			state.behavior_list["P1"].append(result)
			state.scene_list["P1"].append(state_prompt)
			state.state_back["P1"].append(P1_state_back)
			state.state_back["P1_done"].append(P1_done_state_back)
		
		# Build complete action record (using history before the call)
		action_record = {
      		'outputs':response,
			'personality':self.behavior_description,
			"timestep": state.timestep,
			"scene": self.generate_state_prompt(state),
			"conversation_history": conversation_history,
			"think": result.get("Analysis", ""),
			"message": result.get("Message", ""),
			"action": ml_action
		}
		# Save to state's action_records
		if not hasattr(state, 'action_records'):
			state.action_records = {"P0": [], "P1": []}
		state.action_records[f"P{self.agent_index}"].append(action_record)


		if "wait" not in ml_action:
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
		
		print(f"Player {self.agent_index}: {ml_action}")
		self.current_ml_action_steps = 0

		return ml_action



	##################
	'''
	The followings are the Verificator part
	'''
	##################

	def check_current_ml_action_done(self,state):
		"""
		checks if the current ml action is done
		:return: True or False
		"""
		player = state.players[self.agent_index]
		# pot_states_dict = self.mlam.mdp.get_pot_states(state)
		if "pickup" in self.current_ml_action:
			pattern = r"pickup(?:[(]|_)(\w+)(?:[)]|)" # fit both pickup(onion) and pickup_onion
			obj_str = re.search(pattern, self.current_ml_action).group(1)
			return player.has_object() and player.get_object().name == obj_str
		
		elif "fill" in self.current_ml_action:
			return player.held_object.name == 'soup'
		
		elif "put" in self.current_ml_action or "place" in self.current_ml_action:
			return not player.has_object()
		
		elif "deliver" in self.current_ml_action:
			return not player.has_object()
		
		elif "wait" in self.current_ml_action:
			return self.time_to_wait == 0
		elif "message" in self.current_ml_action:
			return True

	def is_ml_action_legal(self, state, ml_action):
		"""
		Check if a given ml_action is legal in the current state.
		This is a refactored version that accepts ml_action as parameter.
		"""
		if ml_action is None:
			return False

		pot_states_dict = self.mdp.get_pot_states(state)
		player = state.players[self.agent_index]
		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			soup_cooking = len(pot_states_dict['cooking']) > 0
			soup_ready = len(pot_states_dict['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)
			cookable_pots = self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			soup_cooking = len(pot_states_dict['onion']['cooking'])+len(pot_states_dict['tomato']['cooking']) > 0
			soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full'] + pot_states_dict["tomato"]['partially_full']
			cookable_pots = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)] # pot has max onions/tomotos

		
		has_onion = False
		has_tomato = False
		has_dish = False
		has_soup = False
		has_object = player.has_object()
		if has_object:
			has_onion = player.get_object().name == 'onion'
			has_tomato = player.get_object().name == 'tomato'
			has_dish = player.get_object().name == 'dish'
			has_soup = player.get_object().name == 'soup'
		empty_counter = self.mdp.get_empty_counter_locations(state)

		original_ml_action = self.current_ml_action
		self.current_ml_action = ml_action

		try:
			if ml_action in ["pickup(onion)", "pickup_onion"]:   
				flag2 = len(self.find_motion_goals(state)) == 0 
				if flag2: 
					return False 
				return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0
			if ml_action in ["pickup(tomato)", "pickup_tomato"]:
				return not has_object and len(self.mdp.get_tomato_dispenser_locations()) > 0
			elif ml_action in ["pickup(dish)", "pickup_dish"]:
				flag2 = len(self.find_motion_goals(state)) == 0 
				if flag2: 
					return False 
				return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
			elif "put_onion_in_pot" in ml_action:
				return has_onion and len(pot_not_full) > 0
			elif "put_tomato_in_pot" in ml_action:
				return has_tomato and len(pot_not_full) > 0
			elif "place_obj_on_counter" in ml_action:
				return has_object and len(empty_counter) > 0
			elif "fill_dish_with_soup" in ml_action:
				return has_dish and (soup_ready or soup_cooking)
			elif "deliver_soup" in ml_action:
				return has_soup
			elif "wait" in ml_action:
				ml_action="wait(1)"
				try:
					wait_time = int(ml_action.split('(')[1].split(')')[0])
					return 0 < wait_time <= 20
				except:
					return False
			elif "message" in ml_action or "set_message" in ml_action:
				return True
			else:
				return False
		finally:
			self.current_ml_action = original_ml_action

	def validate_current_ml_action(self, state):
		"""
		make sure the current_ml_action exists and is valid
		"""
		return self.is_ml_action_legal(state, self.current_ml_action)

	def get_action_condition_description(self, ml_action, state):
		"""
		Get a natural language description of the conditions required for an action.
		Returns a list of condition strings.
		"""
		conditions = []
		player = state.players[self.agent_index]
		has_object = player.has_object()
		
		if ml_action in ["pickup(onion)", "pickup_onion"]:
			conditions.append("I need to have nothing in hand.")
		elif ml_action in ["pickup(tomato)", "pickup_tomato"]:
			conditions.append("I need to have nothing in hand.")
		elif ml_action in ["pickup(dish)", "pickup_dish"]:
			conditions.append("I need to have nothing in hand.")
			pot_states_dict = self.mdp.get_pot_states(state)
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				soup_ready = len(pot_states_dict['ready']) > 0
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
			if soup_ready:
				conditions.append("Need to have a soup ready in <Pot>.")
		elif "put_onion_in_pot" in ml_action:
			conditions.append("I need to have an onion in hand.")
		elif "put_tomato_in_pot" in ml_action:
			conditions.append("I need to have a tomato in hand.")
		elif "place_obj_on_counter" in ml_action:
			conditions.append("I need to have something in hand.")
		elif "fill_dish_with_soup" in ml_action:
			conditions.append("The soup must be ready and cookedI need to have a dish in hand.")
			pot_states_dict = self.mdp.get_pot_states(state)
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				soup_ready = len(pot_states_dict['ready']) > 0
				soup_cooking = len(pot_states_dict['cooking']) > 0
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
				soup_cooking = len(pot_states_dict['onion']['cooking'])+len(pot_states_dict['tomato']['cooking']) > 0
			if soup_ready or soup_cooking:
				conditions.append("Need to have soup ready or cooking in <Pot>.")
		elif "deliver_soup" in ml_action:
			conditions.append("I need to have soup in hand.")
		elif "wait" in ml_action:
			if ml_action == "wait(x)":
				conditions.append("Wait for x timestep(s) and do nothing (x should be between 1 and 20).")
			else:
				try:
					wait_time = int(ml_action.split('(')[1].split(')')[0])
					conditions.append(f"Wait for {wait_time} timestep(s) and do nothing.")
				except:
					conditions.append("Wait for specified timesteps and do nothing.")
		elif "message" in ml_action or "set_message" in ml_action:
			conditions.append(f"Send a message to Player {1-self.agent_index}.")
		
		return conditions

	def enumerate_legal_ml_actions(self, state):
		"""
		Enumerate all legal medium-level actions for the current state.
		Returns a list of dictionaries, each containing:
		- 'action': the action name
		- 'conditions': list of condition descriptions
		"""
		legal_actions = []
		
		# Get all possible actions for this layout and agent
		player_key = f"P{self.agent_index}"
		if self.layout not in env_action:
			return legal_actions
		
		possible_actions = env_action[self.layout].get(player_key, [])
		
		# Check each action for legality
		for action in possible_actions:
			# Normalize action format to match what is_ml_action_legal expects
			normalized_action = action
			if action == "pickup_onion":
				# Check both formats
				if self.is_ml_action_legal(state, "pickup_onion") or self.is_ml_action_legal(state, "pickup(onion)"):
					normalized_action = "pickup_onion"
				else:
					continue
			elif action == "pickup_tomato":
				if self.is_ml_action_legal(state, "pickup_tomato") or self.is_ml_action_legal(state, "pickup(tomato)"):
					normalized_action = "pickup_tomato"
				else:
					continue
			elif action == "pickup_dish":
				if self.is_ml_action_legal(state, "pickup_dish") or self.is_ml_action_legal(state, "pickup(dish)"):
					normalized_action = "pickup_dish"
				else:
					continue
			elif action == "wait(x)":
				# For wait(x), check if wait(1) is legal as a representative
				# The LLM can choose any valid wait time (1-20)
				if self.is_ml_action_legal(state, "wait(1)"):
					normalized_action = "wait(x)"
				else:
					continue
			elif action.startswith("wait("):
				# For specific wait actions, check if the specific wait time is legal
				if self.is_ml_action_legal(state, action):
					normalized_action = action
				else:
					continue
			elif action == "set_message":
				if self.is_ml_action_legal(state, "set_message") or self.is_ml_action_legal(state, "message"):
					normalized_action = "set_message"
				else:
					continue
			else:
				# For other actions, check directly
				if not self.is_ml_action_legal(state, action):
					continue
				normalized_action = action
			
			# Get conditions for this action
			conditions = self.get_action_condition_description(normalized_action, state)
			legal_actions.append({
				'action': normalized_action,
				'conditions': conditions
			})
		
		return legal_actions


	def generate_success_feedback(self, state):
		success_feedback = f"### Controller Validation\nPlayer {self.agent_index} succeeded at {self.current_ml_action}. \n"
		print(success_feedback)  
		if 'wait' not in success_feedback:
			self.planner.add_msg_to_dialog_history({"role": "user", "content": f'Player {self.agent_index} succeeded at {self.current_ml_action}.'})
		
	def generate_failure_feedback(self, state):
		failure_feedback = self.generate_state_prompt(state)
		failure_feedback += f" Player {self.agent_index} failed at {self.current_ml_action}."
		failure_feedback += f" Why did Player {self.agent_index} fail ?"     
		print(f"\n~~~~~~~~ Explainer~~~~~~~~\n{failure_feedback}")  
		failure_message = {"role": "user", "content": failure_feedback}
		self.explainer.current_user_message = failure_message
		failure_explanation = self.explainer.query(self.openai_api_key())
		print(failure_explanation)  
		if "wait" not in failure_explanation or self.layout == 'forced_coodination':
			self.explainer.add_msg_to_dialog_history({"role": "user", "content": failure_feedback})
			self.explainer.add_msg_to_dialog_history({"role": "assistant", "content": failure_explanation})
		self.planner.add_msg_to_dialog_history({"role": "user", "content": failure_explanation}) 


		
	def find_shared_counters(self, state, mlam):  
		counter_dicts = query_counter_states(self.mdp, state) 

		counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
						state.players_pos_and_or[1 - self.agent_index], 
						self.mdp, 
						self.mlam
					)    

		print('counter_list = {}'.format(counter_list))  
		lis = [] 
		for i in counter_list:  
			if counter_dicts[i] == ' ':  
				lis.append(i)       
		available_plans = mlam._get_ml_actions_for_positions(lis)
		return available_plans          

	def find_motion_goals(self, state):
		"""
		Generates the motion goals for the given medium level action.
		:param state:
		:return:
		"""
		am = self.mlam
		motion_goals = []
		player = state.players[self.agent_index]
		pot_states_dict = self.mdp.get_pot_states(state)
		counter_objects = self.mdp.get_counter_objects_dict(
			state, list(self.mdp.terrain_pos_dict["X"])
		)
		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
			motion_goals = am.pickup_onion_actions_new(state, counter_objects, state.players_pos_and_or, self.agent_index) 


		elif self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			motion_goals = am.pickup_tomato_actions(state, counter_objects)
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			motion_goals = am.pickup_dish_actions_new(state, counter_objects , state.players_pos_and_or, self.agent_index)
		elif "put_onion_in_pot" in self.current_ml_action:
			motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
		elif "put_tomato_in_pot" in self.current_ml_action:
			motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
		elif "place_obj_on_counter" in self.current_ml_action:  
			motion_goals = self.find_shared_counters(state, self.mlam)     
			if len(motion_goals) == 0: 
				motion_goals = am.place_obj_on_counter_actions(state)

		elif "start_cooking" in self.current_ml_action:
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				next_order = list(state.all_orders)[0]
				soups_ready_to_cook_key = "{}_items".format(len(next_order.ingredients))
				soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				soups_ready_to_cook = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)]
			only_pot_states_ready_to_cook = defaultdict(list)
			only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
			motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
		elif "fill_dish_with_soup" in self.current_ml_action:
			motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
		elif "deliver_soup" in self.current_ml_action:
			motion_goals = am.deliver_soup_actions()
		elif "wait" in self.current_ml_action:
			motion_goals = am.wait_actions(player)
		else:
			raise ValueError("Invalid action: {}".format(self.current_ml_action))

		motion_goals = [
			mg
			for mg in motion_goals
			if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
				player.pos_and_or, mg
			)
		]

		return motion_goals

	def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
		"""
		For each motion goal, consider the optimal motion plan that reaches the desired location.
		Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
		or rationally), and returns the plan and the corresponding first action on that plan.
		"""

		if self.controller_mode == 'new':
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal_new(
				start_pos_and_or, motion_goals, state
			)
		else: 
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal(
				start_pos_and_or, motion_goals
			)
		return chosen_goal, chosen_goal_action
	
	def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""
		min_cost = np.Inf
		best_action, best_goal = None, None
		for goal in motion_goals:
			action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
				start_pos_and_or, goal
			)
			if plan_cost < min_cost:
				best_action = action_plan[0]
				min_cost = plan_cost
				best_goal = goal
		return best_goal, best_action

	
	def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""   
		min_cost = np.Inf
		best_action, best_goal = None, None
		for goal in motion_goals:   
			action_plan, plan_cost = self.real_time_planner(
				start_pos_and_or, goal, state
			)     
			if plan_cost < min_cost:
				best_action = action_plan
				min_cost = plan_cost
				best_goal = goal     
		if best_action is None: 
      
			if np.random.rand() < 0.5:  
				return None, Action.STAY
			else: 
				return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
		return best_goal, best_action

	def real_time_planner(self, start_pos_and_or, goal, state):   
		terrain_matrix = {
			'matrix': copy.deepcopy(self.mlam.mdp.terrain_mtx), 
			'height': len(self.mlam.mdp.terrain_mtx), 
			'width' : len(self.mlam.mdp.terrain_mtx[0]) 
		}
		other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
		action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, terrain_matrix) 

		return action_plan, plan_cost
	
class ProPlanningAgent(ProAgent):
	def __init__(self, model="gpt-3.5-turbo-0301"):
		super().__init__(model=model)
