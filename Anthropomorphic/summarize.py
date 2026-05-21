import argparse
import json
import os
import random
from string import Template

from openai import OpenAI


def try_send_prompt_random(
    prompt: str,
    prompt_template: str,
    client: OpenAI,
    model: str,
    rng: random.Random,
    max_attempts=5,
    step_reduce=5,
):
    attempt = 0
    lines = prompt.strip().split("\n")
    while attempt <= max_attempts:
        try:
            template = Template(prompt_template)
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
                lines = rng.sample(lines, keep_num)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize player profiles from clustered trajectory actions.")
    parser.add_argument("--data-dir", default="data", help="Directory containing cluster subdirectories with actions.json.")
    parser.add_argument("--output", default="output.jsonl", help="JSONL output path.")
    parser.add_argument("--prompt-file", default="Cook_prompt.txt", help="Prompt template path.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"), help="OpenAI-compatible model name.")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"), help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="API key.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when reducing overlong prompts.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum API retry attempts per cluster.")
    parser.add_argument("--step-reduce", type=int, default=5, help="Number of prompt lines removed after each failed call.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model:
        raise SystemExit("Missing model. Pass --model or set OPENAI_MODEL.")
    if not args.api_base:
        raise SystemExit("Missing API base URL. Pass --api-base or set OPENAI_API_BASE.")

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    rng = random.Random(args.seed)

    if os.path.exists(args.output):
        os.remove(args.output)

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Error: Data directory '{args.data_dir}' does not exist!")

    game_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]

    if not game_dirs:
        raise SystemExit(f"Warning: No subdirectories found in '{args.data_dir}' directory!")

    response_list = []

    for game_dir in sorted(game_dirs):
        actions_file = os.path.join(args.data_dir, game_dir, "actions.json")

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

            response = try_send_prompt_random(
                trajectory_steps,
                prompt_template,
                client,
                args.model,
                rng,
                max_attempts=args.max_attempts,
                step_reduce=args.step_reduce,
            )

            if response is None:
                print(f"Skipping {game_dir}: Failed to get response")
                continue

            profile = extract_profile(response)

            result = {
                "Game": game_dir,
                "Profile": profile
            }

            with open(args.output, "a+", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

            print(f"Processed {game_dir}: {profile[:50]}...")
            response_list.append(response)

        except Exception as e:
            print(f"Error processing {game_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nProcessing complete! Results saved to {args.output}")
    print(f"Total processed: {len(response_list)} games")


if __name__ == "__main__":
    main()
