kill -9 $(lsof -t -i :6320)
python testing_agents/test_symbolic_LLMs.py \
--communication \
--prompt_template_path LLM/prompt_act.csv \
--mode LLMs_act_deepseek-v3 \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6320 \
--lm_id qwen2.5 \
--source openai \
--api_base http://localhost:8007/v1 \
--t 0.7 \
--max_tokens 4096 \
--num_runs 1 \
--num-per-task 2 \
--cot \
--act \
--debug

# kill -9 $(lsof -t -i :6315)
# python testing_agents/test_symbolic_LLMs.py \
# --communication \
# --prompt_template_path LLM/prompt_com.csv \
# --mode LLMs_comm_deepseek-v3 \
# --executable_file ../executable/linux_exec.v2.3.0.x86_64 \
# --base-port 6315 \
# --lm_id deepseek-v3-0324 \
# --source openai \
# --api_base http://ds-v3-0324-671b-16k-int8.api.sii.edu.cn/v1 \
# --t 0.7 \
# --max_tokens 4096 \
# --num_runs 1 \
# --num-per-task 2 \
# --cot \
# --debug

