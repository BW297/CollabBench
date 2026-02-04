for personality in 'Low_Extraversion' 'Medium_Extraversion' 'High_Extraversion' \
                   'Low_Neuroticism' 'Medium_Neuroticism' 'High_Neuroticism' \
                   'Low_Openness' 'Medium_Openness' 'High_Openness' \
                   'Low_Agreeableness' 'Medium_Agreeableness' 'High_Agreeableness' \
                   'Low_Conscientiousness' 'Medium_Conscientiousness' 'High_Conscientiousness'
do
    echo "Running for personality: $personality"
    kill -9 $(lsof -t -i :6317)
    python testing_agents/test_symbolic_LLMs.py \
    --communication \
    --prompt_template_path LLM/prompt_big_5.csv \
    --mode LLMs_act_qwen3-next-235B-A22B_$personality \
    --executable_file ../executable/linux_exec.v2.3.0.x86_64 \
    --base-port 6317 \
    --lm_id qwen3-235b \
    --source openai \
    --api_base https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-a34b599d-c08c-43d1-be23-63a9dc58297c/vscode/b146f810-5a29-4ead-bdff-70160cdf79aa/c5ee7ea0-c154-42b5-b49d-e777624753db/proxy/8000/v1 \
    --t 0.7 \
    --max_tokens 4096 \
    --num_runs 1 \
    --num-per-task 2 \
    --cot \
    --act \
    --big_5 "$personality" \
    --debug
done

