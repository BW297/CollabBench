# for personality in {0..29}; do
#     echo "Running for personality: $personality"
#     kill -9 $(lsof -t -i :6315)
#     python testing_agents/test_symbolic_LLMs.py \
#         --communication \
#         --prompt_template_path LLM/prompt_detail.csv \
#         --mode LLMs_act_qwen2-14B_$personality \
#         --executable_file ../executable/linux_exec.v2.3.0.x86_64 \
#         --base-port 6315 \
#         --lm_id qwen2.5 \
#         --source openai \
#         --api_base http://localhost:8007/v1 \
#         --t 0 \
#         --max_tokens 4096 \
#         --num_runs 5 \
#         --test_task 0 \
#         --num-per-task 2 \
#         --cot \
#         --act \
#         --big_5 "$personality" \
#         --debug
# done

#!/bin/bash

# MODEL="qwen25-72B"
# LM_ID="qwen2.5"
# PORT=6314
# # TASKS=(5 10 16 20 26 30 32 40 49)
# TASKS=(0)
# NUM_PERSONALITIES=30

# for task in "${TASKS[@]}"; do
#     echo "=============================="
#     echo " Running test_task: $task "
#     echo "=============================="

#     for personality in $(seq 10 $((NUM_PERSONALITIES-1))); do
#         echo "---- Running personality: $personality (task=$task) ----"
        
#         # 杀掉端口占用进程
#         pid=$(lsof -t -i :$PORT)
#         if [ ! -z "$pid" ]; then
#             kill -9 $pid
#             sleep 0.5  # 等待端口释放
#         fi

#         python testing_agents/test_symbolic_LLMs.py \
#             --communication \
#             --prompt_template_path LLM/prompt_detail.csv \
#             --mode LLMs_act_${MODEL}_${personality}_task${task} \
#             --executable_file ../executable/linux_exec.v2.3.0.x86_64 \
#             --base-port $PORT \
#             --lm_id $LM_ID \
#             --source openai \
#             --api_base https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/e2fd5ce4-6ceb-4f18-bea3-c9a9be4953e2/c7933cfe-5153-4d0c-98a0-550c6fe7f447/proxy/8042/v1 \
#             --t 0 \
#             --max_tokens 4096 \
#             --num_runs 10 \
#             --test_task $task \
#             --num-per-task 2 \
#             --cot \
#             --act \
#             --big_5 "$personality" \
#             --debug
#     done
# done

MODEL="qwen25-72B"
LM_ID="qwen2.5"
PORT=6317
# TASKS=(5 10 16 20 26 30 32 40 49)
TASKS=(26)
NUM_PERSONALITIES=30

for task in "${TASKS[@]}"; do
    echo "=============================="
    echo " Running test_task: $task "
    echo "=============================="

    for personality in $(seq 15 $((NUM_PERSONALITIES-1))); do
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
            --api_base https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/e2fd5ce4-6ceb-4f18-bea3-c9a9be4953e2/c7933cfe-5153-4d0c-98a0-550c6fe7f447/proxy/8042/v1 \
            --t 0 \
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

MODEL="qwen25-72B"
LM_ID="qwen2.5"
PORT=6317
# TASKS=(5 10 16 20 26 30 32 40 49)
TASKS=(20)
NUM_PERSONALITIES=30

for task in "${TASKS[@]}"; do
    echo "=============================="
    echo " Running test_task: $task "
    echo "=============================="

    for personality in $(seq 17 $((NUM_PERSONALITIES-1))); do
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
            --api_base https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/e2fd5ce4-6ceb-4f18-bea3-c9a9be4953e2/c7933cfe-5153-4d0c-98a0-550c6fe7f447/proxy/8042/v1 \
            --t 0 \
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

# ===== 参数 =====
MODEL="qwen25-72B"
LM_ID="qwen2.5"
PORT=6315
TASKS=(16 32 40 49)
# 16 20
# TASKS=(0)
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
            --api_base https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/e2fd5ce4-6ceb-4f18-bea3-c9a9be4953e2/c7933cfe-5153-4d0c-98a0-550c6fe7f447/proxy/8042/v1 \
            --t 0 \
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