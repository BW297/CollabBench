set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONBREAKPOINT=0
export PROAGENT_DEBUG=0
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="agent_system/environments/env_package/proagent:${PYTHONPATH}"

num_cpus_per_env_worker=0.5 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=4
val_data_size=10
group_size=8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
# 1. 禁用可能引起冲突的特性
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Multi-layout training: each env_id corresponds to a different layout
# Layouts: cramped_room, asymmetric_advantages, coordination_ring, counter_circuit, forced_coordination
# With train_data_size=5 and group_size=1, each layout gets 1 env

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="YOUR_LOG_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_proagent_${TIMESTAMP}.log"
echo "Log file: $LOG_FILE"

TRAIN_MODEL_PATH="YOUR_TRAIN_MODEL_PATH"
COOK_URL="YOUR_TRAIN_URL"
COOK_MODEL_ID="YOUR_TRAIN_MODEL_ID"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$TRAIN_MODEL_PATH  \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=proagent \
    'env.proagent.layouts=[cramped_room,asymmetric_advantages,coordination_ring,counter_circuit,forced_coordination]' \
    env.proagent.horizon=150 \
    env.proagent.p0=RL \
    env.proagent.p1=ProAgent \
    env.proagent.base_url=$COOK_URL \
    env.proagent.lm_id=$COOK_MODEL_ID \
    env.proagent.sampling_parameters.t=0.7 \
    env.proagent.sampling_parameters.max_tokens=4096 \
    env.proagent.sampling_parameters.top_p=1.0 \
    env.proagent.sampling_parameters.n=1 \
    env.proagent.debug=false \
    env.proagent.sampling_parameters.debug=False \
    env.proagent.sampling_parameters.prompt_level=l2-ap_merged \
    env.proagent.sampling_parameters.belief_revision=false \
    env.proagent.sampling_parameters.retrival_method=recent_k \
    env.proagent.sampling_parameters.K=1 \
    env.proagent.sampling_parameters.using_big_5=false \
    env.proagent.sampling_parameters.big_five=Extraversion \
    env.proagent.sampling_parameters.level=Low \
    env.seed=0 \
    env.max_steps=150 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_proagent' \
    trainer.experiment_name='gigpo_qwen2.5_7b_proagent' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False $@ 2>&1 | tee "$LOG_FILE"
