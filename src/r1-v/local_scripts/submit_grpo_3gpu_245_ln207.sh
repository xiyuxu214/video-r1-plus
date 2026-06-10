#!/bin/bash
# Run from local_scripts on login node: bash submit_grpo_3gpu_245_ln207.sh
# Edit only the exec line GPU args until yhbatch accepts (do not add gpu #SBATCH inside run_grpo_3gpu_245.sh).
cd "$(dirname "$0")"
exec yhbatch --gres=gpu:3 run_grpo_3gpu_245.sh
# If line above fails, try ONE of these instead (comment out others):
# exec yhbatch -G 3 run_grpo_3gpu_245.sh
# exec yhbatch --gpus-per-node=3 run_grpo_3gpu_245.sh
# exec yhbatch --gres=dcu:3 run_grpo_3gpu_245.sh
