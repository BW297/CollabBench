MODEL="qwen25-72B"
LM_ID="qwen2.5"
PORT=6390
TASKS=(0 5 10 16 20 26 30 32 40 49)
NUM_PERSONALITIES=30

for task in "${TASKS[@]}"; do
    echo "=============================="
    echo " Running test_task: $task "
    echo "=============================="

    for personality in $(seq 0 $((NUM_PERSONALITIES-1))); do
        echo "---- Running personality: $personality (task=$task) ----"
        
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
            --api_base https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/0a81fd12-0266-4ce8-a10f-d6d9efc76274/1a04a5e7-46d4-4b01-9609-cc0d4bbe71ac/proxy/8007/v1 \
            --t 1.0 \
            --max_tokens 4096 \
            --num_runs 10 \
            --test_task $task \
            --num-per-task 2 \
            --cot \
            --act \
            --big_5 "$personality" \
            --debug
    done
done