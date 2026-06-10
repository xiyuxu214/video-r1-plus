#!/bin/bash
# 在登录节点执行（与 run_grpo_3gpu_245 同套路）:
#   cd /path/to/Video-R1-main/src/r1-v/local_scripts
#   chmod +x submit_robust_grpo_hydra_2gpu_yhbatch.sh run_robust_grpo_hydra_2gpu.sh
#
# 1) 先编辑 run_robust_grpo_hydra_2gpu.sh 顶部的 #SBATCH -D、VENV、MODEL 等路径。
# 2) 再执行本脚本申请 2 张 GPU 并提交作业。

cd "$(dirname "$0")"

# 二选一：能通的那一行保留，其余注释掉（与 submit_grpo_3gpu_245_ln207.sh 一致）
exec yhbatch --gres=gpu:2 ./run_robust_grpo_hydra_2gpu.sh
# exec yhbatch -G 2 ./run_robust_grpo_hydra_2gpu.sh
# exec yhbatch --gpus-per-node=2 ./run_robust_grpo_hydra_2gpu.sh
# exec yhbatch --gres=dcu:2 ./run_robust_grpo_hydra_2gpu.sh
