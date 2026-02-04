kill -9 $(lsof -t -i :6315)
python testing_agents/test_symbolic_LLMs.py \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--mode LLMs_comm_qwen2.5-72B_test_v2 \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6315 \
--lm_id qwen25 \
--source openai \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--cot \
--debug

