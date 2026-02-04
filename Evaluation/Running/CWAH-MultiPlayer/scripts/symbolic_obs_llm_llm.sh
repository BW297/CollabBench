# kill -9 $(lsof -t -i :6315)
# python testing_agents/test_symbolic_LLMs.py \
# --communication \
# --prompt_template_path LLM/prompt_com.csv \
# --mode LLMs_comm_qwen2.5-72B_test_v2 \
# --executable_file ../executable/linux_exec.v2.3.0.x86_64 \
# --base-port 6315 \
# --lm_id qwen25 \
# --source openai \
# --t 0.7 \
# --max_tokens 256 \
# --num_runs 1 \
# --num-per-task 2 \
# --cot \
# --debug

# ===== 参数 =====
MODEL="qwen25-72B"
LM_ID="qwen2.5"
PORT=6312
TASKS=(0 5 10 16 20 26 30 32 40 49)
NUM_PERSONALITIES=30

for task in "${TASKS[@]}"; do
    echo "=============================="
    echo " Running test_task: $task "
    echo "=============================="

    # for personality in $(seq 29 $((NUM_PERSONALITIES-1))); do
    #     echo "---- Running personality: $personality (task=$task) ----"
        
    # 杀掉端口占用进程
    pid=$(lsof -t -i :$PORT)
    if [ ! -z "$pid" ]; then
        kill -9 $pid
        sleep 0.5  # 等待端口释放
    fi

    python testing_agents/test_symbolic_LLMs.py \
        --communication \
        --prompt_template_path LLM/prompt_detail.csv \
        --mode LLMs_act_${MODEL}_${personality}_task${task} \
        --executable_file ../executable/linux_exec.v2.3.0.x86_64 \
        --base-port $PORT \
        --lm_id $LM_ID \
        --source openai \
        --api_base https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/e2fd5ce4-6ceb-4f18-bea3-c9a9be4953e2/c7933cfe-5153-4d0c-98a0-550c6fe7f447/proxy/8042/v1 \
        --t 0 \
        --max_tokens 4096 \
        --num_runs 10 \
        --test_task $task \
        --num-per-task 2 \
        --cot \
        --debug
done
