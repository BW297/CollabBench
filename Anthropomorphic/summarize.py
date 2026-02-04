from openai import OpenAI
import json
import os
import numpy as np
from string import Template

cook_prompt_file = "Cook_prompt.txt"
cwah_prompt_file = "CWAH_prompt.txt"
with open(cook_prompt_file, "r", encoding="utf-8") as f:
    PROMPT = f.read()

model = "YOUR_MODEL"
client = OpenAI(base_url="YOUR_DEPLOYMENT_URL", api_key="YOUR_API_KEY")

import random

def try_send_prompt_random(prompt: str, max_attempts=5, step_reduce=5):
    attempt = 0
    lines = prompt.strip().split("\n")
    while attempt <= max_attempts:
        try:
            template = Template(PROMPT)
            full_prompt = template.safe_substitute(trajectory_steps=prompt)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                top_p=0.85,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt+1}: API call failed: {e}")
            attempt += 1
           
            if len(lines) > step_reduce:
                keep_num = len(lines) - step_reduce
                lines = random.sample(lines, keep_num)
                prompt = "\n".join(lines)
            else:
                print("Prompt too short, skipping this cluster")
                return None
    return None

def format_trajectory_steps(player_data, steps_per_example=3):
    if not player_data or len(player_data) == 0:
        return ""
    
    trajectory_str = ""
    example_id = 1
    
    for i in range(0, len(player_data), steps_per_example):
        chunk_data = player_data[i:i+steps_per_example]
        
        if not chunk_data:
            continue
        
        step_str = ""
        for step_idx, step_data in enumerate(chunk_data):
            think = step_data.get('think', '')
            action = step_data.get('action', '')
            
            if action == 'set_message' or 'set_message' in action.lower():
                message = step_data.get('message', '')
                if message:
                    action = f"{action}: {message}"
            
            step_str += f"Step {step_idx + 1}- Analysis: {think} Plan: {action}\n"
        
        trajectory_str += f"Example ID {example_id}. {step_str}"
        example_id += 1
    
    return trajectory_str

def extract_profile(text):
    if "Profile:" not in text:
        return "NO_PROFILE"
    
    profile_part = text.split("Profile:", 1)[1].strip()
    
    profile_part = profile_part.replace("```", "").strip()
    
    if profile_part.startswith("(") and profile_part.endswith(")"):
        profile_part = profile_part[1:-1].strip()
    
    return profile_part

data_dir = "data"
output_file = "output.json"

if os.path.exists(output_file):
    os.remove(output_file)

if not os.path.exists(data_dir):
    print(f"Error: Data directory '{data_dir}' does not exist!")
    exit(1)

game_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

if not game_dirs:
    print(f"Warning: No subdirectories found in '{data_dir}' directory!")
    exit(1)

response_list = []

for game_dir in sorted(game_dirs):
    actions_file = os.path.join(data_dir, game_dir, "actions.json")
    
    if not os.path.exists(actions_file):
        print(f"Warning: actions.json not found in {game_dir}, skipping...")
        continue
    
    try:
        with open(actions_file, "r", encoding="utf-8") as f:
            trajectory_data = json.load(f)
        
        if not isinstance(trajectory_data, list):
            print(f"Warning: {game_dir}/actions.json is not a list, skipping...")
            continue
        
        if len(trajectory_data) == 0:
            print(f"Warning: No trajectory data in {game_dir}, skipping...")
            continue
        
        trajectory_steps = format_trajectory_steps(trajectory_data, steps_per_example=3)
        
        if not trajectory_steps:
            print(f"Warning: Empty trajectory for {game_dir}, skipping...")
            continue
        
        response = try_send_prompt_random(trajectory_steps)
        
        if response is None:
            print(f"Skipping {game_dir}: Failed to get response")
            continue
        
        profile = extract_profile(response)
        
        result = {
            "Game": game_dir,
            "Profile": profile
        }
        
        with open(output_file, "a+", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
        
        print(f"Processed {game_dir}: {profile[:50]}...")
        response_list.append(response)
        
    except Exception as e:
        print(f"Error processing {game_dir}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\nProcessing complete! Results saved to {output_file}")
print(f"Total processed: {len(response_list)} games")
