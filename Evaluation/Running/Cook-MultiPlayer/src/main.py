import time
import datetime
import os
import json
import copy
from argparse import ArgumentParser
import numpy as np
from rich import print as rprint
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*") # ignore "Unable to register cuBLAS factory" due to use tf-CPU
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

# import pkg_resources
# VERSION = pkg_resources.get_distribution("overcooked_ai").version
import importlib_metadata
VERSION = importlib_metadata.version("overcooked_ai")
print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action


from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent
from logger import Logger
def main(variant):

    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    using_big_5 = variant['using_big_5']
    big_five = variant['big_five']
    level = variant['level']

    mode = variant['mode']
    
    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == '0.0.1':
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    
    p0_algo = variant['p0']
    p1_algo = variant['p1']
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")


    start_time = time.time()
    results = []
    episode_step=-1

    for i in range(episode):  
        episode_step=i

        agents_list = []
        for alg in [p0_algo, p1_algo]:
            if alg == "ProAgent":
                assert variant['running_model']!=None, print(f'you should choose a gpt model')
                print(f"\n----Use {variant['running_model']}----\n")
                running_model = variant['running_model']
                retrival_method = variant['retrival_method']
                K = variant['K']
                prompt_level = variant['prompt_level']
                belief_revision = variant['belief_revision']
                agent = make_agent(alg, mdp, layout, model=running_model, 
                                   prompt_level=prompt_level, 
                                   belief_revision=belief_revision, 
                                   retrival_method=retrival_method, K=K,
                                   using_big_5=variant['using_big_5'],
                                   big_5=variant['big_five'],
                                   level=variant['level'],
                                   agent_url=variant.get('agent_url'),
                                   agent_model=variant.get('agent_model'),
                                   agent_api_key=variant.get('agent_api_key'),
                                   player_url=variant.get('player_url'),
                                   player_model=variant.get('player_model'),
                                   player_api_key=variant.get('player_api_key'))
            elif alg == "BC":
                agent = make_agent(alg, mdp, layout, seed_id=i)
            else:
                agent = make_agent(alg, mdp, layout)
            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()

        env.state.action_records={"P0":[],"P1":[]}

        
        r_total = 0

        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                print(env.mdp.state_string(s_t).replace('ø', 'o'))   
                # chat_prompt = self.generate_state_prompt(state) + self.generate_chat_prompt() + " Based on the above scenario description, what would you like to say to your teammate?"
                a_t = team.joint_action(s_t) 
                
                print(f"\n-----------Controller-----------\n")    
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")

                last_action_records = copy.deepcopy(s_t.action_records) if hasattr(s_t, 'action_records') and s_t.action_records else {"P0":[],"P1":[]}

                obs, reward, done, env_info = env.step(a_t)

                env.state.action_records=last_action_records

                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills) 

                r_total += reward
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')

                final_action_records = copy.deepcopy(last_action_records)

            ## finish one episode
            if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
                print(f"\n================\n")
                try: # ProAgent id = 0
                    print(f"P1's real behavior: {team.agents[0].teammate_ml_actions_dict}")
                    print(f"The infered P1's intention: {team.agents[0].teammate_intentions_dict}")
                except: # ProAgent id = 1
                    print(f"P0's real behavior: {team.agents[1].teammate_ml_actions_dict}")
                    print(f"The infered P0's intention: {team.agents[1].teammate_intentions_dict}")
                print(f"\n================\n")

            folder_act_path = "actions"
            act_file = f"{folder_act_path}/actions_{episode}_{horizon}_{layout}_{running_model}_{prompt_level}_{retrival_method}_{K}_using_big_5_{using_big_5}_{level}/step_{episode_step+1}"
            os.makedirs(act_file,exist_ok=True)

            actions_data = {
                "0": final_action_records.get("P0", []),
                "1": final_action_records.get("P1", [])
            }
            with open(act_file+"/actions.json", "w", encoding="utf-8") as f:
                json.dump(actions_data, f, indent=4, ensure_ascii=False)

        elif mode == 'demo':
            pass
         
        print(f"Episode {episode_step+1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)

        
   
    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")



    result_dict = {
        "input": variant,
        "raw_results": results,
        "mean_result": int(np.mean(results)),
        'cost_time': end_time - start_time
    }
    for (k,v) in result_dict.items():
        print(f'{k}: {v}')
    


    folder_result_path = "results"
        
    os.makedirs(folder_result_path,exist_ok=True)

    if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
        json_file = f"{folder_result_path}/results_{episode}_{horizon}_{layout}_{running_model}_{prompt_level}_{retrival_method}_{K}_using_big_5_{variant['using_big_5']}_{variant['level']}.json"
    else:
        json_file = f"{folder_result_path}/results_{episode}_{horizon}_{layout}.json"
    with open(json_file, "w") as f:
        json.dump(result_dict, f, indent=4)



    
if __name__ == '__main__':

    '''
    python main.py --layout cramped_room --p0 Greedy --p1 Greedy --horizon 100
    python main.py --layout cramped_room --p0 ProAgent --p1 BC --horizon 400 -pl l2-ap
    '''
    parser = ArgumentParser(description='OvercookedAI Experiment')

    # these are basis parses
    parser.add_argument('--layout', '-l', type=str, default='cramped_room', choices=['cramped_room', 'asymmetric_advantages', 'coordination_ring', 'forced_coordination', 'counter_circuit'])
    parser.add_argument('--p0',  type=str, default='Greedy', choices=['ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay', 'Human'], help='Algorithm for P0 agent 0')
    parser.add_argument('--p1', type=str, default='Greedy', choices=['ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay', 'Human'], help='Algorithm for P1 agent 1')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon steps in one game')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')

    parser.add_argument('--using_big_5', type=boolean_argument, default=True, help="Enable big 5 function")
    parser.add_argument('--level', type=int, default=0, help='Level of big 5')    

    parser.add_argument('--running_model', type=str, default='qwen2.5', choices=['text-davinci-003', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314', 'Qwen2-1.5B-Instruct', 'qwen25', 'deepseek-v3-0324', 'Qwen2','Qwen3', 'qwen2.5-7B', 'qwen2.5-72b-instruct', 'qwen3-8B'], help='Number of episodes')
    parser.add_argument('--prompt_level', '-pl', type=str, default='l2-ap', choices=['l1-p', 'l2-ap', 'l3-aip'], help="'l1-p': make plans directly without CoT; 'l2-ap': plans with analysis; 'l3-aip': plans with analysis and intention.")
    parser.add_argument('--belief_revision', '-br', type=boolean_argument, default=False, help='whether we use belief_revision or not')
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'], help='Use similarity-based(BERT, CLIP) retrieval or retrieve recent K history in dialog.')
    parser.add_argument('--K', type=int, default=1, help="The number of dialogues you want to retrieve.")

    # API configuration for agent (agent_index=0) and player (agent_index=1)
    parser.add_argument('--agent_url', type=str, default=None, help='API URL for agent (agent_index=0)')
    parser.add_argument('--agent_model', type=str, default=None, help='Model name for agent (agent_index=0)')
    parser.add_argument('--agent_api_key', type=str, default=None, help='API key for agent (agent_index=0)')
    parser.add_argument('--player_url', type=str, default=None, help='API URL for player (agent_index=1)')
    parser.add_argument('--player_model', type=str, default=None, help='Model name for player (agent_index=1)')
    parser.add_argument('--player_api_key', type=str, default=None, help='API key for player (agent_index=1)')

    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'demo'], help='exp mode run step-by-step, demo mode run via traj')                                
    parser.add_argument('--save', type=boolean_argument, default=True, help='Whether save the result')
    parser.add_argument('--log_dir', type=str, default=None, help='dir to save result')
    parser.add_argument('--debug', type=boolean_argument, default=True, help='debug mode')


    args = parser.parse_args()
    variant = vars(args)

    with Logger():
        start_time = time.time()
        main(variant)
        end_time = time.time()
        print(f"\n=======Finshed all=========\n")
        print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
