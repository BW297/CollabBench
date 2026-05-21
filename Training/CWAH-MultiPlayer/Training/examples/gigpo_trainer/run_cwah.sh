set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE="offline"
export HF_ENDPOINT=https://hf-mirror.com

num_cpus_per_env_worker=0.5 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=4
val_data_size=10
group_size=8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
timestamp=$(date +%Y%m%d_%H%M%S)

TRAIN_MODEL_PATH="YOUR_TRAIN_MODEL_PATH"
CWAH_URL="YOUR_TRAIN_URL"
CWAH_MODEL_ID="YOUR_TRAIN_MODEL_ID"
# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$TRAIN_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.use_token_length_penalty=False \
    actor_rollout_ref.actor.token_length_penalty_coef=0.0002 \
    actor_rollout_ref.actor.use_interactivity_reward=True \
    actor_rollout_ref.actor.interactivity_reward_coef=0.25 \
    actor_rollout_ref.actor.use_message_reward=True \
    actor_rollout_ref.actor.message_reward_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=cwah \
    env.cwah.base_url=$CWAH_URL \
    env.cwah.lm_id=$CWAH_MODEL_ID \
    env.cwah.sampling_parameters.max_tokens=4096 \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_cwah' \
    trainer.experiment_name='gigpo_qwen25_7b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=3 \
    trainer.test_freq=5 \
    trainer.total_epochs=1000 \
    trainer.val_before_train=False $@ 2>&1 | tee train_${timestamp}.log
