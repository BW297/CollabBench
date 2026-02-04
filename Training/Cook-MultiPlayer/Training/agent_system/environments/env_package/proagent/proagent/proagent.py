import itertools, os, json, re
from collections import defaultdict
import numpy as np
import pkg_resources
import sys 
import copy 
from .modules import Module
import sys
from string import Template
import json
import csv
with open("actions.json", 'r') as file:
    action_list_01 = json.load(file)
# 通过环境变量控制调试输出
_PROAGENT_DEBUG = os.environ.get("PROAGENT_DEBUG", "0").lower() in ("1", "true", "yes")

# Add lib to path for overcooked_ai import
base_dir = os.path.dirname(os.path.dirname(__file__))
lib_path = os.path.join(base_dir, 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import find_path 
from overcooked_ai_py.planning.search import get_intersect_counter 
from overcooked_ai_py.planning.search import query_counter_states 
np.random.seed(2)

# Layout name mapping for prompt files (prompt files use OLD_LAYOUTS names)
LAYOUT_TO_PROMPT_NAME = {
    "counter_circuit": "random3",
    "forced_coordination": "random0",
    "cramped_room": "simple",
    "coordination_ring": "random1",
    "asymmetric_advantages": "unident_s"
}

# Set up paths for prompts and openai key
base_dir = os.path.dirname(os.path.dirname(__file__))
prompts_base = os.path.join(os.path.dirname(base_dir), '..', '..', 'prompts', 'proagent_prompts')
if not os.path.exists(prompts_base):
    prompts_base = os.path.join(base_dir, '..', '..', 'prompts', 'proagent_prompts')
PROMPT_DIR = prompts_base

# Try to find openai_key.txt in various locations
cwd = os.getcwd()
possible_key_locations = [
    os.path.join(cwd, "openai_key.txt"),
    os.path.join(base_dir, "openai_key.txt"),
    os.path.join(os.path.dirname(base_dir), "openai_key.txt"),
]
openai_key_file = None
for loc in possible_key_locations:
    if os.path.exists(loc):
        openai_key_file = loc
        break
if openai_key_file is None:
    openai_key_file = os.path.join(cwd, "openai_key.txt")  # Default fallback

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
		# 强制结束标志：某些分支会临时置 True 以结束当前高阶动作（必须提前初始化，避免 AttributeError）
		self._force_ml_action_done = False

		self.openai_api_keys = []
		self.load_openai_keys()
		self.key_rotation = True

	def load_openai_keys(self):
		with open(openai_key_file, "r") as f:
			context = f.read()
		self.openai_api_keys = context.split('\n')

	def openai_api_key(self):
		if self.key_rotation:
			self.update_openai_key()
		return self.openai_api_keys[0]

	def update_openai_key(self):
		self.openai_api_keys.append(self.openai_api_keys.pop(0))

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
			using_big_5=False,
			big_5=None,
			level=None,
			base_url=None,
			api_key=None,
			worker_id=None,  # Worker ID for logging
			profile=None,    # 人格配置
	):
		super().__init__(model=model)
		self.base_url = base_url
		self.api_key = api_key
		self.worker_id = worker_id  # Store worker_id for logging
		self.profile = profile  #人格配置
		# Debug: print model and base_url
		if base_url:
			worker_prefix = f"[Worker-{worker_id}] " if worker_id is not None else ""
			if _PROAGENT_DEBUG:
				print(f"{worker_prefix}[ProAgent] Initialized with model={model}, base_url={base_url}")

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
		self._cached_message = None  # Cache message from merged prompt format
		# 强制结束标志：agent0 在“高阶动作不合法”时置 True，用于让上层结束本轮高阶动作
		self._force_ml_action_done = False

		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.pot_id_to_pos = []

		self.using_big_5 = using_big_5
		self.big_5 = big_5
		self.level = level
		self.trace_list = []
		if self.using_big_5:
			# Read from cook.csv: coo/Running/Cook-MultiPlayer/src/prompts/cook.csv
			# Find coo directory by traversing up from current file
			current_path = os.path.abspath(__file__)
			coo_dir = None
			parts = current_path.split(os.sep)
			for i in range(len(parts) - 1, -1, -1):
				if parts[i] == 'coo':
					coo_dir = os.sep.join(parts[:i+1])
					break
			# If coo not found, try relative path from Training root
			if coo_dir is None:
				# Fallback: assume we're in Training/Cook-MultiPlayer/Training/... and go up to coo
				current_dir = os.path.dirname(os.path.abspath(__file__))
				# Go up: proagent -> proagent -> env_package -> environments -> agent_system -> Training -> Cook-MultiPlayer -> Training -> coo
				coo_dir = os.path.normpath(os.path.join(current_dir, "../../../../../../.."))
			cook_csv_path = os.path.normpath(os.path.join(
				coo_dir, "Running", "Cook-MultiPlayer", "src", "prompts", "cook.csv"
			))
			try:
				self.big_5_desp_1 = []
				with open(cook_csv_path, "r", encoding="utf-8") as f:
					reader = csv.DictReader(f)
					for row in reader:
						item = {
							'Task': row.get('Task', '').strip(),
							'Profile': row.get('Profile', '').strip(),
							'Cluster': row.get('Cluster', '').strip(),
							'Examples': ''  # cook.csv doesn't have Examples, set empty
						}
						if item['Task'] and item['Profile']:
							self.big_5_desp_1.append(item)
			except FileNotFoundError:
				if _PROAGENT_DEBUG:
					print(f"Warning: Could not find cook.csv at {cook_csv_path}, using empty list")
				self.big_5_desp_1 = []
			except Exception as e:
				if _PROAGENT_DEBUG:
					print(f"Warning: Error reading cook.csv at {cook_csv_path}: {e}, using empty list")
				self.big_5_desp_1 = []
			self.big_5_desp = {}
			for item in self.big_5_desp_1:
				value_dict={}
				value_dict['Profile']=item['Profile']
				value_dict['Examples']=item.get('Examples', '')  # Default to empty string if missing
				key = f"{item['Task']}_{item['Cluster']}"
				self.big_5_desp[key] = value_dict

		self.layout_prompt = self.generate_layout_prompt()


	def set_mdp(self, mdp):
		self.mdp = mdp

	def create_gptmodule(self, module_name, file_type='txt', retrival_method='recent_k', K=10, batch_client=None):
		worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
		if _PROAGENT_DEBUG:
			print(f"{worker_prefix}[ProAgent] 🔧 初始化 GPT 模块: {module_name}")    

		# prompt_file = os.path.join(PROMPT_DIR, self.model, module_name, self.layout+f'_{self.agent_index}.'+file_type)

		if "gpt" in self.model or "text-davinci" in self.model:
			model_name = "gpt"
		elif "claude" in self.model:
			model_name = "claude"
		else:
			model_name= "gpt"
	
		if module_name == "planner":
			# Use mapped layout name for prompt files
			prompt_layout_name = LAYOUT_TO_PROMPT_NAME.get(self.layout, self.layout)
			if not self.using_big_5:
				# Support merged prompt format (l2-ap_merged)
				if self.prompt_level == "l2-ap_merged":
					prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, self.prompt_level, f'{prompt_layout_name}_{self.agent_index}.{file_type}')
				else:
					prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, self.prompt_level, f'{prompt_layout_name}_{self.agent_index}.{file_type}')
			else:
				prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, self.prompt_level+"_big_5", f'{prompt_layout_name}_{self.agent_index}.{file_type}')
		elif module_name == "explainer":
			prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, f'player{self.agent_index}.{file_type}')
		elif module_name == "chat":
			if not self.using_big_5:
				prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, f'CHAT_player_{self.agent_index}.{file_type}')
			else:
				prompt_file = os.path.join(PROMPT_DIR, model_name, f"{module_name}_big_5", f'CHAT_player_{self.agent_index}.{file_type}')
		else:
			raise Exception(f"Module {module_name} not supported.")

		# Try to find prompt file if it doesn't exist at expected location
		if not os.path.exists(prompt_file):
			# Use mapped layout name for alternative locations
			prompt_layout_name = LAYOUT_TO_PROMPT_NAME.get(self.layout, self.layout)
			# Try alternative locations
			alt_locations = [
				os.path.join(os.path.dirname(PROMPT_DIR), model_name, module_name, self.prompt_level if module_name == "planner" else "", f'{prompt_layout_name}_{self.agent_index}.{file_type}'),
				os.path.join(PROMPT_DIR, f'{prompt_layout_name}_{self.agent_index}.{file_type}'),
			]
			for alt_loc in alt_locations:
				if os.path.exists(alt_loc):
					prompt_file = alt_loc
					break

		# print(prompt_file)

		try:
			with open(prompt_file, "r") as f:
				raw_content = f.read()
				if file_type == 'json':
					messages = json.load(f)
				elif file_type == 'txt':
					if self.using_big_5 and self.agent_index==0 and module_name in ["planner","chat"]:
						# from rich import print as rprint
						# rprint("[red][OPENAI PRINT][/red]:", "okokokokokokokokokokokokokokokokok")
						string_tem = Template(raw_content)
	     # ,personality_exp=self.big_5_desp[f"{self.layout}_{str(self.level)}"]['Examples']
						message = string_tem.substitute(personality_def=self.big_5_desp[f"{self.layout}_{str(self.level)}"]['Profile'])
						# self.behavior_description = self.big_5_desp[self.level]['Profile']
						worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
						if _PROAGENT_DEBUG:
							print(f"{worker_prefix}[ProAgent] 📋 行为描述 (Behavior Description): {self.behavior_description}")
						messages = [{"role": "system", "content": message}]
					else:
						# For l2-ap_merged planner prompts, extract candidate actions and store template
						if module_name == "planner" and self.prompt_level == "l2-ap_merged":
							# Extract candidate actions from skill descriptions in the prompt file
							# Look for lines like "    - pickup(onion):" or "    - put_onion_in_pot():"
							candidate_actions = []
							for action in action_list_01[prompt_layout_name][f"{self.agent_index}"]:
								candidate_actions.append(action['action'])
							# Always include set_message() if not already present (it's always available)
							if 'set_message()' not in candidate_actions and 'set_message' not in candidate_actions:
								candidate_actions.append('set_message()')
							# Store candidate actions for this layout+agent
							if not hasattr(self, '_candidate_ml_actions'):
								self._candidate_ml_actions = []
							self._candidate_ml_actions = candidate_actions
							# Store the raw template (with $allowed_actions placeholder)
							if not hasattr(self, '_planner_prompt_template'):
								self._planner_prompt_template = None
							self._planner_prompt_template = raw_content
							if _PROAGENT_DEBUG:
								worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
								print(f"{worker_prefix}[ProAgent] 📋 提取候选动作 (Extracted candidate actions for {self.layout}, agent {self.agent_index}): {candidate_actions}")
							if len(candidate_actions) == 0:
								worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
								if _PROAGENT_DEBUG:
									print(f"{worker_prefix}[ProAgent] ⚠️  警告: 未找到候选动作 (Warning: No candidate actions found)")
						messages = [{"role": "system", "content": raw_content}]
				else:
					worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
					if _PROAGENT_DEBUG:
						print(f"{worker_prefix}[ProAgent] ⚠️  不支持的文件格式 (Unsupported file format)")
					messages = [{"role": "system", "content": ""}]
		except FileNotFoundError:
			worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
			if _PROAGENT_DEBUG:
				print(f"{worker_prefix}[ProAgent] ⚠️  警告: 提示词文件未找到 (Warning: Prompt file not found): {prompt_file}, 使用空提示词")
			messages = [{"role": "system", "content": ""}]
		except Exception as e:
			worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
			if _PROAGENT_DEBUG:
				print(f"{worker_prefix}[ProAgent] ❌ 加载提示词文件错误 (Error loading prompt file): {prompt_file}")
				print(f"   错误信息 (Error): {e}")
			messages = [{"role": "system", "content": ""}]
		
		return Module(messages, self.model, retrival_method, K, base_url=self.base_url, api_key=self.api_key, batch_client=batch_client)

	def reset(self):
		self.planner.reset()
		self.explainer.reset()
		self.prev_state = None
		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self._force_ml_action_done = False
		self.possible_motion_goals = None
		self.current_timestep = 0
		self.teammate_ml_actions_dict = {}
		self.teammate_intentions_dict = {}
		self._cached_message = None  # Reset cached message
		if self.using_big_5:
			self.behavior_description = self.big_5_desp[f"{self.layout}_{str(self.level)}"]['Profile']
		# self.teammate_chat_messages = {"0": [], "1": []}

	def set_agent_index(self, agent_index, batch_client=None):
		self.agent_index = agent_index
		self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K, batch_client=batch_client)
		self.chat = self.create_gptmodule("chat", retrival_method=self.retrival_method, K=self.K, batch_client=batch_client)
		self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=self.K, batch_client=batch_client)

		worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
		if _PROAGENT_DEBUG:
			print(f"{worker_prefix}[ProAgent] 📝 Planner 系统提示词 (System Prompt):")
			print("-"*80)
			print(self.planner.instruction_head_list[0]['content'][:500] + "..." if len(self.planner.instruction_head_list[0]['content']) > 500 else self.planner.instruction_head_list[0]['content'])
			print("-"*80 + "\n")

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
		layout_prompt = layout_prompt[:-1] + ".\n"
		return layout_prompt

	def generate_kitchen_prompt(self, state):
		
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

		return kitchen_state_prompt
  
	def generate_state_prompt(self, state):
		ego = state.players[self.agent_index]
		teammate = state.players[1 - self.agent_index]

		time_prompt = f"Scene {state.timestep}: "
		# time_prompt = f"Scene: "
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

		if self.layout == 'forced_coordination': 
			teammate_state_prompt = ""
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

	def action(self, state):

		start_pos_and_or = state.players_pos_and_or[self.agent_index]

		# only use to record the teammate ml_action, 
		# if teammate finish ml_action in t-1, it will record in s_t, 
		# otherwise, s_t will just record None,
		# and we here check this information and store it into proagent
		self.current_timestep = state.timestep
		if state.ml_actions[1-self.agent_index] != None:
			self.teammate_ml_actions_dict[str(self.current_timestep-1)] = state.ml_actions[1-self.agent_index]

		# if current ml action does not exist, generate a new one
		if self.current_ml_action is None:
			if self.agent_index != 0: #如果不是agent0，则生成新的ml动作
				self.current_ml_action = self.generate_ml_action(state)

		# if the current ml action is in process, Player{self.agent_index} done, else generate a new one
		if self.current_ml_action_steps > 0:
			current_ml_action_done = self.check_current_ml_action_done(state)
			if current_ml_action_done:
				# generate a new ml action
				if self.agent_index != 0:
					self.generate_success_feedback(state)
					self.current_ml_action = self.generate_ml_action(state)

		count = 0
		worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
		while not self.validate_current_ml_action(state):
				
			if self.agent_index == 0:
				if _PROAGENT_DEBUG:
					print(f"{worker_prefix}[Player 0] ⚠️  动作无效 (Invalid action): {self.current_ml_action}, 重新生成中...")
				# 判空保护：首次无效时 P0 的回溯列表可能为空，避免 [-1]/pop 越界
				if state.scene_list.get("P0") and len(state.scene_list["P0"]) > 0:
					state.fail_back.setdefault("P0", []).append(state.scene_list["P0"][-1])
				if state.behavior_list.get("P0") and len(state.behavior_list["P0"]) > 0:
					state.behavior_list["P0"].pop()
				if state.scene_list.get("P0") and len(state.scene_list["P0"]) > 0:
					state.scene_list["P0"].pop()
				if state.state_back.get("P0") and len(state.state_back["P0"]) > 0:
					state.state_back["P0"].pop()
				self.generate_failure_feedback(state)
				# 对于 agent0：一旦检测到高阶动作不合法，标记"本轮高阶动作结束"（上层会据此停止继续执行）
				self._force_ml_action_done = True
				
				break
			else:
				if _PROAGENT_DEBUG:
					print(f"{worker_prefix}[Player 1] ⚠️  动作无效 (Invalid action): {self.current_ml_action}, 重新生成中...")
				# 判空保护：首次无效时 P1 的回溯列表可能为空，避免 [-1]/pop 越界
				if hasattr(state, "fail_back") and isinstance(state.fail_back, dict):
					if "P1" not in state.fail_back:
						state.fail_back["P1"] = []
				if hasattr(state, "scene_list") and isinstance(state.scene_list, dict):
					if "P1" not in state.scene_list:
						state.scene_list["P1"] = []
				if hasattr(state, "behavior_list") and isinstance(state.behavior_list, dict):
					if "P1" not in state.behavior_list:
						state.behavior_list["P1"] = []
				if hasattr(state, "state_back") and isinstance(state.state_back, dict):
					if "P1" not in state.state_back:
						state.state_back["P1"] = []
					if "P1_done" not in state.state_back:
						state.state_back["P1_done"] = []

				if hasattr(state, "scene_list") and isinstance(state.scene_list, dict) and len(state.scene_list.get("P1", [])) > 0:
					state.fail_back["P1"].append(state.scene_list["P1"][-1])
					state.scene_list["P1"].pop()
				else:
					# 没有可回溯的 scene，记录空占位
					state.fail_back["P1"].append("")

				if hasattr(state, "behavior_list") and isinstance(state.behavior_list, dict) and len(state.behavior_list.get("P1", [])) > 0:
					state.behavior_list["P1"].pop()
				if hasattr(state, "state_back") and isinstance(state.state_back, dict) and len(state.state_back.get("P1", [])) > 0:
					state.state_back["P1"].pop()
				if hasattr(state, "state_back") and isinstance(state.state_back, dict) and len(state.state_back.get("P1_done", [])) > 0:
					state.state_back["P1_done"].pop()
			# agent0 强制结束：避免 AttributeError，并且只触发一次
			
			self.trace = False
			self.generate_failure_feedback(state)
			self.current_ml_action = self.generate_ml_action(state)
			
			count += 1
			if count > 1:
				if self.agent_index == 0:
					if _PROAGENT_DEBUG:
						print(f"{worker_prefix}[Player 0] ⚠️  动作无效 (Invalid action): {self.current_ml_action}, 重新生成中...")
					# 判空保护：避免 pop 越界
					if len(state.behavior_list.get("P0", [])) > 0:
						state.behavior_list["P0"].pop()
					# 根因修复：behavior_list/state_back 有 push 时，scene_list 也必须同步 push/pop
					if hasattr(state, "scene_list") and isinstance(state.scene_list, dict) and len(state.scene_list.get("P0", [])) > 0:
						state.scene_list["P0"].pop()
					if len(state.state_back.get("P0", [])) > 0:
						state.state_back["P0"].pop()
				else:
					if _PROAGENT_DEBUG:
						print(f"{worker_prefix}[Player 1] ⚠️  动作无效 (Invalid action): {self.current_ml_action}, 重新生成中...")
					# 判空保护：避免 pop 越界
					if len(state.behavior_list.get("P1", [])) > 0:
						state.behavior_list["P1"].pop()
					# 根因修复：behavior_list/state_back 有 push 时，scene_list 也必须同步 push/pop
					if hasattr(state, "scene_list") and isinstance(state.scene_list, dict) and len(state.scene_list.get("P1", [])) > 0:
						state.scene_list["P1"].pop()
					if len(state.state_back.get("P1", [])) > 0:
						state.state_back["P1"].pop()
					if len(state.state_back.get("P1_done", [])) > 0:
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
				# 根因修复：这里会 push 行为与回溯信息，必须同步 push scene_list，保证后续回退分支不对空 pop
				scene_snapshot = self.generate_state_prompt(state)
				if self.agent_index == 0:
					state.behavior_list["P0"].append(result)	
					if hasattr(state, "scene_list") and isinstance(state.scene_list, dict):
						state.scene_list.setdefault("P0", []).append(scene_snapshot)
					state.state_back["P0"].append(P0_state_back)
				else:
					state.behavior_list["P1"].append(result)
					if hasattr(state, "scene_list") and isinstance(state.scene_list, dict):
						state.scene_list.setdefault("P1", []).append(scene_snapshot)
					state.state_back["P1"].append(P1_state_back)
					state.state_back["P1_done"].append(P1_done_state_back)
				self.time_to_wait = 1
				break
		
		self.trace = True 
		# agent0 强制结束：用于“高阶动作不合法”时让上层直接跳过 env.step
		# 这里必须返回一个合法的低阶动作（不能返回 None），否则 Overcooked 会报 Illegal action None
		if self.agent_index == 0 and getattr(self, "_force_ml_action_done", False):
			self._force_ml_action_done = False
			return Action.STAY, {"ml_action_completed": True, "ml_action_invalid": True}
		if "wait" in self.current_ml_action:
			self.current_ml_action_steps += 1
			self.time_to_wait -= 1
			lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
			chosen_action =lis_actions[np.random.randint(0,len(lis_actions))]
			chosen_action = Action.STAY
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				self.prev_state = state
				return chosen_action, {}
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				self.prev_state = state
				return chosen_action,{}
		elif "message" in self.current_ml_action:
			state.steps[str(self.agent_index)]=0
			self.current_ml_action_steps += 1
			lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
			chosen_action = lis_actions[np.random.randint(0, len(lis_actions))]
			chosen_action = Action.STAY
			
			# If using merged format, use cached message from planner response
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			if self.prompt_level == "l2-ap_merged" and hasattr(self, '_cached_message') and self._cached_message:
				response = self._cached_message
				if _PROAGENT_DEBUG:
					print(f"{worker_prefix}[Player {self.agent_index}] 💬 使用合并提示词中的消息: {response}")
			else:
				# Original format: call chat module separately
				chat_prompt = self.generate_state_prompt(state) +"\n\n"+ self.generate_team_chat_prompt(state.haschat_other, state.chat_list) + f"\n\nBased on the above scenario description, what would you like to say to Player {1-self.agent_index} ?\n"
				if _PROAGENT_DEBUG:
					print("\n" + "-"*80)
					print(f"{worker_prefix}[Player {self.agent_index}] 💬 生成聊天消息 (Generating Chat Message)")
					print("-"*80)
					print(f"提示词: {chat_prompt[:200]}..." if len(chat_prompt) > 200 else f"提示词: {chat_prompt}")
					print("-"*80)
				self.chat.current_user_message={"role": "user", "content": chat_prompt}
				response = self.chat.query(key=self.openai_api_key(), stop='Scene', trace = self.trace)
				if _PROAGENT_DEBUG:
					print(f"{worker_prefix}[Player {self.agent_index}] 💬 生成的消息: {response}\n")
			
			if self.agent_index == 0:
				state.chat_list["P0"].append({"scene": self.generate_state_prompt(state),"content": response})
			else:
				state.chat_list["P1"].append({"scene": self.generate_state_prompt(state),"content": response})
			# 直接追加完整的 set_message 记录，不依赖 behavior_list[-1]
			if self.agent_index == 0:
				state.behavior_list.setdefault("P0", []).append({"Plan": f"set_message: {response}"})
			else:
				state.behavior_list.setdefault("P1", []).append({"Plan": f"set_message: {response}"})

			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				self.prev_state = state
				return chosen_action, {}
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				self.prev_state = state
				return chosen_action, {}
		else:
			possible_motion_goals = self.find_motion_goals(state)    
			current_motion_goal, chosen_action = self.choose_motion_goal(
				start_pos_and_or, 
				possible_motion_goals, 
				state
			)
			if _PROAGENT_DEBUG:
				print(f"current_motion_goal: {current_motion_goal}, chosen_action: {chosen_action}")
		# if "wait" in self.current_ml_action: 
		# 	print(f'current motion goal for P{self.agent_index} is wait') 
		# else: 
		# 	if current_motion_goal is None: 
		# 		current_motion_goal = 'None' 
		# 	print(f'current motion goal for P{self.agent_index} is {current_motion_goal}') 


		if self.auto_unstuck and chosen_action != Action.INTERACT:
			if _PROAGENT_DEBUG:
				print(f"auto_unstuck: {self.auto_unstuck}, chosen_action: {chosen_action}")
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
				# 判空保护：避免 pop 越界
				if len(state.behavior_list.get("P0", [])) > 0:
					state.behavior_list["P0"].pop()
				# 根因修复：behavior_list/state_back 有 push 时，scene_list 也必须同步 push/pop
				if hasattr(state, "scene_list") and isinstance(state.scene_list, dict) and len(state.scene_list.get("P0", [])) > 0:
					state.scene_list["P0"].pop()
				if len(state.state_back.get("P0", [])) > 0:
					state.state_back["P0"].pop()
			else:
				# 判空保护：避免 pop 越界
				if len(state.behavior_list.get("P1", [])) > 0:
					state.behavior_list["P1"].pop()
				# 根因修复：behavior_list/state_back 有 push 时，scene_list 也必须同步 push/pop
				if hasattr(state, "scene_list") and isinstance(state.scene_list, dict) and len(state.scene_list.get("P1", [])) > 0:
					state.scene_list["P1"].pop()
				if len(state.state_back.get("P1", [])) > 0:
					state.state_back["P1"].pop()
				if len(state.state_back.get("P1_done", [])) > 0:
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
			# 根因修复：这里会 push 行为与回溯信息，必须同步 push scene_list，保证后续回退分支不对空 pop
			scene_snapshot = self.generate_state_prompt(state)
			if self.agent_index == 0:
				state.behavior_list["P0"].append(result)
				state.scene_list.setdefault("P0", []).append(scene_snapshot)
				state.state_back["P0"].append(P0_state_back)
			else:
				state.behavior_list["P1"].append(result)
				state.scene_list.setdefault("P1", []).append(scene_snapshot)
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
			return chosen_action,{}
	
	def generate_team_chat_prompt(self, haschat, chat_list, recent_k=1):
		chat_prompt = "Previous chat messages from your teammate: "

		other_index = 1 - self.agent_index
		msgs = chat_list.get(f'P{other_index}') or []
		if msgs:
			msgs = msgs[-recent_k:] if len(msgs) >= recent_k else msgs
			for msg in msgs:
				chat_prompt += f"Player {other_index} said: '{msg['content']}'. "
			# haschat[f"P{other_index}"]=False
		else:
			chat_prompt += "No previous chat messages. "

		return chat_prompt

	# def generate_chat_prompt(self, chat_list, recent_k=1):
	# 	chat_prompt = "Previous chat messages: "
	# 	p0_msgs = chat_list.get("P0") or []
	# 	p1_msgs = chat_list.get("P1") or []

	# 	i0, i1 = len(p0_msgs) - 1, len(p1_msgs) - 1
	# 	count = 0

	# 	if i0 < 0 and i1 < 0:
	# 		chat_prompt += "No previous chat messages. "
	# 	else:
	# 		while count < recent_k and (i0 >= 0 or i1 >= 0):
	# 			if i0 >= 0 and count < recent_k:
	# 				msg = p0_msgs[i0]
	# 				chat_prompt += f"In Scene: {msg['scene']}\nPlayer 0 said: '{msg['content']}'. "
	# 				i0 -= 1
	# 				count += 1
	# 			if i1 >= 0 and count < recent_k:
	# 				msg = p1_msgs[i1]
	# 				chat_prompt += f"In Scene: {msg['scene']}\nPlayer 1 said: '{msg['content']}'. "
	# 				i1 -= 1
	# 				count += 1
	# 	return chat_prompt


	def parse_ml_action(self, response, agent_index): 
		# Support merged format (l2-ap_merged) with <action> tags
		if self.prompt_level == "l2-ap_merged":
			# Try to extract from <action> tags first
			action_pattern = r'<action>\s*(.+?)\s*</action>'
			match = re.findall(action_pattern, response, re.DOTALL)
			if match:
				action_string = match[-1].strip()
				# Remove quotes if present
				action_string = action_string.strip('"').strip("'")
			else:
				# Fallback to old format
				if agent_index == 0:
					pattern = r'Plan for Player 0:\s*(.+)'
				elif agent_index == 1:
					pattern = r'Plan for Player 1:\s*(.+)'
				else:
					raise ValueError("Unsupported agent index.")
				match = re.findall(pattern, response, re.DOTALL)
				if match:
					action_string = match[-1]
				else:
					action_string = response
		else:
			# Original format
			if agent_index == 0:
				pattern = r'Plan for Player 0:\s*(.+)'
			elif agent_index == 1:
				pattern = r'Plan for Player 1:\s*(.+)'
			else:
				raise ValueError("Unsupported agent index.")

			match = re.findall(pattern, response, re.DOTALL)
			if match:
				action_string = match[-1]
			else:
				# raise Exception("please check the query")
				action_string = response
				# print("please check the query")

		# Parse the response to get the medium level action string
		try: 
			# For merged format, action_string might already be the full action
			if self.prompt_level == "l2-ap_merged":
				# Check if it contains parentheses (like "pickup(onion)" or "wait(1)")
				if '(' in action_string:
					ml_action = action_string.split('(')[0].strip()
				else:
					ml_action = action_string.split()[0] if action_string.split() else action_string
			else:
				ml_action = action_string.split()[0]
		except Exception as e: 
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			# if _PROAGENT_DEBUG:
			print(f"{worker_prefix}[Player {self.agent_index}] ⚠️  解析动作失败 (Failed to parse action): {e}")
			print(f"  原始字符串 (Original string): {action_string}")
			action_string = 'wait(1)'
			ml_action = action_string 

		# Parse action_string to extract the actual action
		# For merged format, action_string might be like "pickup(onion)" or "set_message()"
		# For original format, action_string might be like "pickup_onion" or "set_message"
		if "place" in action_string.lower():
			ml_action = "place_obj_on_counter"
		elif "pick" in action_string.lower():
			if "onion" in action_string.lower():
				ml_action = "pickup_onion"
			elif "tomato" in action_string.lower():
				ml_action = "pickup_tomato"
			elif "dish" in action_string.lower():
				ml_action = "pickup_dish"
		elif "put" in action_string.lower():
			if "onion" in action_string.lower():
				ml_action = "put_onion_in_pot"
			elif "tomato" in action_string.lower():
				ml_action = "put_tomato_in_pot"
		elif "fill" in action_string.lower():   
			ml_action = "fill_dish_with_soup"
		elif "deliver" in action_string.lower():
			ml_action = "deliver_soup"
		elif "message" in action_string.lower():
			ml_action = "set_message"
		elif "wait" not in action_string.lower():
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
				# 这里可以改一下试试 
				self.time_to_wait = max(3, parse_wait_string(action_string))
			else: 
				self.time_to_wait = parse_wait_string(action_string)    
			# print(ml_action) 
			# print(self.time_to_wait) 
			
			ml_action = f"wait({self.time_to_wait})"

		else:
			pass
		
		# aviod to generate two skill, eg, Plan for Player 0: "deliver_soup(), pickup(onion)".
		if "," in ml_action:
			ml_action = ml_action.split(',')[0].strip()

		            
		return ml_action    


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
		# Update planner prompt with state-specific allowed actions (for l2-ap_merged)
		if self.prompt_level == "l2-ap_merged":
			self._update_planner_prompt_for_state(state)
		
		if self.prompt_level == "l3-aip" and self.belief_revision:
			belief_prompt = self.generate_belief_prompt()
		else:
			belief_prompt = ''
		state_prompt = belief_prompt + self.generate_trace_prompt(state.behavior_list, state.scene_list) + "\n\n" + self.generate_state_prompt(state) +"\n\n"+ self.generate_team_chat_prompt(state.haschat, state.chat_list, 1)

		state_message = {"role": "user", "content": state_prompt}
		self.planner.current_user_message = state_message
		
		# 手动设置 cache_list 以便打印（模拟 query 方法中的逻辑）
		# 注意：这里需要临时设置 K=0 来获取所有历史记录用于打印
		original_K = self.planner.K
		self.planner.cache_list = self.planner.get_cache()
		self.planner.K = original_K  # 恢复原始值
		
		# ========== 打印完整提示词 ==========
		if _PROAGENT_DEBUG:
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			print("\n" + "="*80)
			print(f"{worker_prefix}[Player {self.agent_index}] 📝 发送给 LLM 的完整提示词 (Full Prompt to LLM)")
			print("="*80)
			
			# 打印系统提示词
			if self.planner.instruction_head_list:
				system_prompt = self.planner.instruction_head_list[0].get('content', '')
				if system_prompt:
					print("\n【系统提示词 (System Prompt)】")
					print("-" * 80)
					print(system_prompt)
					print("-" * 80)
			
			# 打印历史对话（如果有）
			if self.planner.cache_list:
				print("\n【历史对话 (Dialog History)】")
				print("-" * 80)
				for msg in self.planner.cache_list:
					role = msg.get('role', 'unknown')
					content = msg.get('content', '')
					# 只显示前200个字符，避免太长
					content_preview = content[:200] + "..." if len(content) > 200 else content
					print(f"{role}: {content_preview}")
				print("-" * 80)
			
			# 打印当前用户消息
			print("\n【当前用户消息 (Current User Message)】")
			print("-" * 80)
			print(state_prompt)
			print("-" * 80)
			
			print("="*80 + "\n")

		response = self.planner.query(key=self.openai_api_key(), stop='Scene', trace = self.trace)
		
		if 'wait' not in response:
			self.planner.add_msg_to_dialog_history(state_message) 
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
		
		# ========== 打印 LLM 回答 ==========
		if _PROAGENT_DEBUG:
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			print("\n" + "="*80)
			print(f"{worker_prefix}[Player {self.agent_index}] 🤖 LLM 的完整回答 (LLM Response)")
			print("="*80)
			print(response)
			print("="*80 + "\n")
		
		result = {"Analysis": "", "Plan": "", "Message": ""}

		# Extract Analysis/Think section
		if self.prompt_level == "l2-ap_merged":
			# Extract from <think> tags (matching prompt template)
			think_pattern = r'<think>\s*(.+?)\s*</think>'
			think_match = re.findall(think_pattern, response, re.DOTALL)
			if think_match:
				result["Analysis"] = think_match[-1].strip()
			# Extract message from <message> tags
			message_pattern = r'<message>\s*(.+?)\s*</message>'
			message_match = re.findall(message_pattern, response, re.DOTALL)
			if message_match:
				result["Message"] = message_match[-1].strip()
		else:
			# Original format: extract Analysis segment
			pattern = r"Analysis:\s*(.*?)(?=Plan for Player \d+:|$)"
			analysis_match = re.findall(pattern, response, re.S)
			if analysis_match:
				result["Analysis"] = analysis_match[-1]

		if _PROAGENT_DEBUG:
			print("\n===== Parser =====\n")
		## specific for prompt need intention
		if self.prompt_level == "l3-aip":
			generated_intention = self.parse_ml_action(response, 1-self.agent_index)
			self.teammate_intentions_dict[str(self.current_timestep)] = generated_intention
			if _PROAGENT_DEBUG:
				print(f"Intention for Player {1 - self.agent_index}: {generated_intention}")  
			# if str(self.current_timestep) in self.teammate_intentions_dict:   
			# 	self.teammate_intentions_dict[str(self.current_timestep)].append(generated_intention)
			# else: 
			# 	self.teammate_intentions_dict[str(self.current_timestep)] = [] 
			# 	self.teammate_intentions_dict[str(self.current_timestep)].append(generated_intention) 

		ml_action = self.parse_ml_action(response, self.agent_index)
		result["Plan"] = ml_action
		
		# Store message if using merged format
		if self.prompt_level == "l2-ap_merged" and result.get("Message"):
			# Store the message for later use when set_message is selected
			self._cached_message = result["Message"]
		else:
			self._cached_message = None
		
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


		if "wait" not in ml_action:
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
		
		# ========== 动作生成完成 ==========
		if _PROAGENT_DEBUG:
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			print(f"{worker_prefix}[Player {self.agent_index}] ✅ 生成的动作: {ml_action}")
			print("")
		
		self.current_ml_action_steps = 0
		if ml_action == "set_message":
			state.haschat[f"P{self.agent_index}"] = True
			state.haschat_other[f"P{self.agent_index}"] = True

		return ml_action

	def generate_trace_prompt(self, behavior_list, scene_list, K=3):
		trace_prompt=""
		# if self.agent_index == 0:
		# 	key = "P0"
		# 	prefix = "Player 0's previous actions are as follows:\n "
		# else:
		# 	key = "P1"
		# 	prefix = "Player 1's previous actions are as follows:\n "
		# scenes = scene_list.get(key, [])
		# behaviours = behavior_list.get(key, [])
		# if not scenes or not behaviours:
		# 	return prefix + "No previous records available.\n"
		# records = list(zip(scenes, behaviours))

		# recent = records[-K:]

		# trace_prompt = prefix
		# start_index = len(records) - len(recent)
		# for i, (scene, beh) in enumerate(recent, start=start_index):
		# 	# trace_prompt += f"Scene {i}----\n"
		# 	# trace_prompt += f"Scene Description: {scene}"
		# 	trace_prompt += f"{beh.get('Plan', '<no Plan field>')}\n"

		return trace_prompt


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
		# 外部强制终止：用于 agent0 在“高阶动作不合法”时结束本轮高阶动作（只生效一次）
		if getattr(self, "_force_ml_action_done", False):
			self._force_ml_action_done = False
			return True

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

	def validate_current_ml_action(self, state):
		"""
		make sure the current_ml_action exists and is valid
		"""
		if self.current_ml_action is None:
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


		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:   

			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False 
			return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0
		if self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			return not has_object and len(self.mdp.get_tomato_dispenser_locations()) > 0
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False 
			return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
		elif "put_onion_in_pot" in self.current_ml_action:
			return has_onion and len(pot_not_full) > 0
		elif "put_tomato_in_pot" in self.current_ml_action:
			return has_tomato and len(pot_not_full) > 0
		elif "place_obj_on_counter" in self.current_ml_action:
			return has_object and len(empty_counter) > 0
		elif "fill_dish_with_soup" in self.current_ml_action:
			return has_dish and (soup_ready or soup_cooking)
		elif "deliver_soup" in self.current_ml_action:
			return has_soup
		elif "wait" in self.current_ml_action:
			return 0 < int(self.current_ml_action.split('(')[1][:-1]) <= 20
		elif "message" in self.current_ml_action:
			return True

	def _compute_allowed_ml_actions(self, state):
		"""
		Filter candidate ML actions for the current state using validate_current_ml_action.
		Returns a list of action strings that are valid for the current state.
		"""
		if not hasattr(self, '_candidate_ml_actions') or not self._candidate_ml_actions:
			# Fallback: return empty list if no candidates extracted
			if _PROAGENT_DEBUG:
				worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
				print(f"{worker_prefix}[ProAgent] ⚠️  无候选动作，返回空列表 (No candidate actions, returning empty list)")
			return []
		
		allowed_actions = []
		original_ml_action = self.current_ml_action
		original_time_to_wait = self.time_to_wait
		
		for candidate in self._candidate_ml_actions:
			try:
				# Temporarily set the candidate as current_ml_action for validation
				self.current_ml_action = candidate
				# Handle wait actions specially
				if "wait" in candidate.lower():
					# Parse wait duration
					wait_match = re.search(r'wait\((\d+)\)', candidate)
					if wait_match:
						self.time_to_wait = int(wait_match.group(1))
					else:
						self.time_to_wait = 1
				else:
					self.time_to_wait = 0
				
				# Validate the action
				if self.validate_current_ml_action(state):
					allowed_actions.append(candidate)
			except Exception as e:
				# If validation fails for any reason, skip this candidate
				if _PROAGENT_DEBUG:
					worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
					print(f"{worker_prefix}[ProAgent] ⚠️  验证动作失败 (Validation failed for {candidate}): {e}")
				continue
			finally:
				# Restore original values
				self.current_ml_action = original_ml_action
				self.time_to_wait = original_time_to_wait
		
		# Safety: if no actions are allowed, include set_message() if it's in candidates (it's always valid)
		if len(allowed_actions) == 0:
			if 'set_message()' in self._candidate_ml_actions:
				allowed_actions.append('set_message()')
			elif _PROAGENT_DEBUG:
				worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
				print(f"{worker_prefix}[ProAgent] ⚠️  警告: 当前状态无合法动作，使用空列表 (Warning: No valid actions for current state)")
		
		return allowed_actions

	def _update_planner_prompt_for_state(self, state):
		"""
		Update the planner system prompt with state-specific allowed actions.
		Replaces $allowed_actions placeholder with the filtered list of valid actions.
		"""
		if not hasattr(self, '_planner_prompt_template') or self._planner_prompt_template is None:
			# Not using l2-ap_merged or template not loaded, skip
			return
		
		if not hasattr(self, 'planner') or self.planner is None:
			return
		
		# Compute allowed actions for current state
		allowed_actions = self._compute_allowed_ml_actions(state)
		
		# Format the allowed actions list as a comma-separated string
		if len(allowed_actions) > 0:
			allowed_actions_str = ', '.join(allowed_actions)
		else:
			# Fallback: if somehow no actions are allowed, use a safe default
			allowed_actions_str = 'set_message()'
			if _PROAGENT_DEBUG:
				worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
				print(f"{worker_prefix}[ProAgent] ⚠️  无合法动作，使用默认 (No valid actions, using default)")
		
		# Replace the placeholder in the template
		updated_prompt = self._planner_prompt_template.replace('$allowed_actions', allowed_actions_str)
		if self.agent_index ==1:
			updated_prompt = updated_prompt.replace('$profile', self.profile)
			print("Agent1 profile:--",self.profile)
		
		# Update the planner's instruction_head_list
		if self.planner.instruction_head_list and len(self.planner.instruction_head_list) > 0:
			self.planner.instruction_head_list[0]['content'] = updated_prompt
			# if _PROAGENT_DEBUG:
			worker_prefix = f"[Worker-{self.worker_id}] " if hasattr(self, 'worker_id') and self.worker_id is not None else ""
			print(f"{worker_prefix}[ProAgent] ✅ 更新提示词，合法动作 ({len(allowed_actions)}): {allowed_actions_str[:100]}...")

	def generate_success_feedback(self, state):
		success_feedback = f"### Controller Validation\nPlayer {self.agent_index} succeeded at {self.current_ml_action}. \n"
		if _PROAGENT_DEBUG:
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			print(f"{worker_prefix}[Player {self.agent_index}] ✅ 动作执行成功: {self.current_ml_action}")
		if 'wait' not in success_feedback:
			self.planner.add_msg_to_dialog_history({"role": "user", "content": f'Player {self.agent_index} succeeded at {self.current_ml_action}.'})
		
	def generate_failure_feedback(self, state):
		failure_feedback = self.generate_state_prompt(state)
		failure_feedback += f" Player {self.agent_index} failed at {self.current_ml_action}."
		failure_feedback += f" Why did Player {self.agent_index} fail ?"
		
		if _PROAGENT_DEBUG:
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			print("\n" + "="*80)
			print(f"{worker_prefix}[Player {self.agent_index}] ❌ 动作执行失败 (Action Failed)")
			print("="*80)
			print(f"失败的动作 (Failed Action): {self.current_ml_action}")
			print("\n发送给解释器的提示词 (Prompt to Explainer):")
			print("-"*80)
			print(failure_feedback)
			print("-"*80)
		
		failure_message = {"role": "user", "content": failure_feedback}
		self.explainer.current_user_message = failure_message
		failure_explanation = self.explainer.query(self.openai_api_key())
		
		if _PROAGENT_DEBUG:
			print("\n解释器的回答 (Explainer Response):")
			print("-"*80)
			print(failure_explanation)
			print("="*80 + "\n")
		
		if "wait" not in failure_explanation or self.layout == 'forced_coodination':
			self.explainer.add_msg_to_dialog_history({"role": "user", "content": failure_feedback})
			self.explainer.add_msg_to_dialog_history({"role": "assistant", "content": failure_explanation})
		self.planner.add_msg_to_dialog_history({"role": "user", "content": failure_explanation}) 

	##################
	'''
	The followings are the Controller part almost inherited from GreedyHumanModel class
	'''
	##################	
		
	def find_shared_counters(self, state, mlam):  
		counter_dicts = query_counter_states(self.mdp, state) 

		counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
						state.players_pos_and_or[1 - self.agent_index], 
						self.mdp, 
						self.mlam
					)    

		if _PROAGENT_DEBUG:
			worker_prefix = f"[Worker-{self.worker_id}] " if self.worker_id is not None else ""
			print(f"{worker_prefix}[Player {self.agent_index}] 📍 共享计数器列表: {counter_list}")  
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
		min_cost = np.inf
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
		min_cost = np.inf
		best_action, best_goal = None, None
		for goal in motion_goals:   
			action_plan, plan_cost = self.real_time_planner(
				start_pos_and_or, goal, state
			)     
			if _PROAGENT_DEBUG:
				print(f"action_plan: {action_plan}, plan_cost: {plan_cost}")
			if plan_cost < min_cost:
				best_action = action_plan
				min_cost = plan_cost
				best_goal = goal 
		if _PROAGENT_DEBUG:
			print(f"best_action: {best_action}, best_goal: {best_goal}")
		if best_action is None: 
			# print('\n\n\nBlocking Happend, executing default path\n\n\n')
			# print('current position = {}'.format(start_pos_and_or)) 
			# print('goal position = {}'.format(motion_goals))        
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

