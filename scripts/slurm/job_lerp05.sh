#!/bin/bash
#SBATCH --job-name=rlinf_lerp05
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.log
#SBATCH --cpus-per-task=8

apptainer exec --nv \
  --bind ~/scratch/Mar15-RLinf:/workspace/RLinf \
  --bind ~/scratch/.cache:/root/.cache \
  ~/scratch/containers_rlinf_isaaclab_openpi_gr00t/rlinf_isaaclab_openpi_gr00t.sif \
  bash -c "
    source /usr/local/bin/switch_env openpi
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    export AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    cd /workspace/RLinf/isaac_sim && source ./setup_conda_env.sh
    cd /workspace/RLinf
    ray stop --force || true
    export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
    export DISPLAY=""

    HYDRA_FULL_ERROR=1 TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python /workspace/RLinf/examples/embodiment/train_embodied_agent_amp.py \
      --config-path /workspace/RLinf/examples/embodiment/config/ \
      --config-name isaaclab_franka_stack_cube_amp_ppo_openpi_pi05 \
      runner.logger.log_path=/workspace/RLinf/results \
      runner.logger.project_name=rlinf \
      runner.logger.experiment_name=isaaclab_ppo_openpi_pi05_amp_lerp05 \
      'runner.logger.logger_backends=[tensorboard,wandb]' \
      actor.micro_batch_size=4 \
      actor.global_batch_size=32 \
      runner.save_interval=10 \
      runner.val_check_interval=5 \
      algorithm.update_epoch=1 \
      algorithm.rollout_epoch=1 \
      algorithm.kl_beta=0.01 \
      algorithm.entropy_bonus=0.001 \
      actor.optim.lr=2e-6 \
      env.train.auto_reset=True \
      rollout.enable_offload=False \
      amp.hdf5_path=/workspace/RLinf/datasets/rendered_200_states.hdf5 \
      amp.save_rewards=false \
      amp.task_reward_lerp=0.5 \
      env.eval.video_cfg.video_base_dir=/workspace/RLinf/results/video/isaaclab_ppo_openpi_pi05_amp_lerp05 \
      rollout.model.model_path=/workspace/RLinf/outputs/RLinf-pi05-SFT-Stack-cube \
      actor.model.model_path=/workspace/RLinf/outputs/RLinf-pi05-SFT-Stack-cube
  "
