for personality in 'Low_Extraversion' 'Medium_Extraversion' 'High_Extraversion' \
                   'Low_Neuroticism' 'Medium_Neuroticism' 'High_Neuroticism' \
                   'Low_Openness' 'Medium_Openness' 'High_Openness' \
                   'Low_Agreeableness' 'Medium_Agreeableness' 'High_Agreeableness' \
                   'Low_Conscientiousness' 'Medium_Conscientiousness' 'High_Conscientiousness'
do
    echo "Running for personality: $personality"
    kill -9 $(lsof -t -i :6316)
    python testing_agents/test_symbolic_LLMs.py \
    --communication \
    --prompt_template_path LLM/prompt_big_5.csv \
    --mode LLMs_act_deepseek-v3_4096_$personality \
    --executable_file ../executable/linux_exec.v2.3.0.x86_64 \
    --base-port 6316 \
    --lm_id deepseek-v3-0324 \
    --source openai \
    --api_base http://ds-v3-0324-671b-16k-int8.api.sii.edu.cn/v1 \
    --t 0.7 \
    --max_tokens 4096 \
    --num_runs 1 \
    --num-per-task 2 \
    --cot \
    --act \
    --big_5 "$personality" \
    --debug
done

