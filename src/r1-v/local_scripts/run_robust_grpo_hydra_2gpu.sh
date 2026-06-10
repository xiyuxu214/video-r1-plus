#!/bin/bash
# 双卡 Robust GRPO（Hydra）作业体：由 yhbatch 提交，GPU 数量在 submit 脚本里申请。
# 集群若禁止在脚本里写 GPU 相关 #SBATCH，请保持本文件不含 --gres / -G。
#
# 提交（在 login 节点、于 local_scripts 目录）:
#   bash submit_robust_grpo_hydra_2gpu_yhbatch.sh
#
# 文件须 LF 换行；路径请改成你集群上的真实路径。

#SBATCH -J r1v-robust-2gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o ./outputs/yhbatch-robust-%j.out
#SBATCH -e ./outputs/yhbatch-robust-%j.err
# 工作目录：r1-v 根目录（含 src/open_r1、config）
#SBATCH -D /path/to/Video-R1-main/src/r1-v

set -euo pipefail

# ============ 按集群修改 ============
export R1V="${R1V:-$(pwd)}"
export OUTDIR="${OUTDIR:-${R1V}/outputs/robust-grpo-2gpu-${SLURM_JOB_ID:-local}}"
export VENV="${VENV:-/path/to/your/conda/envs/video-r1}"
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
# Hydra 实验名：完整训练用 base；消融示例见下方注释
export HYDRA_EXPERIMENT="${HYDRA_EXPERIMENT:-base}"

module purge 2>/dev/null || true
# module load CUDA/12.x   # 按集群打开

mkdir -p "$OUTDIR" "${R1V}/outputs"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export PATH="$VENV/bin:$PATH"

# 作业若已分配 2 张卡，通常可见 cuda:0,1；勿与 yhbatch 申请数量不一致
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-1} % 20000 + 1))}"
echo "JOB=${SLURM_JOB_ID:-na} MASTER_PORT=$MASTER_PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OUTDIR=$OUTDIR"

cd "${R1V}/src/open_r1"

# 双进程 = 2 张训练卡（无 vLLM 独立卡；若你改用 grpo.py + vLLM，需 3 卡以上）
exec torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  train_robust_grpo_hydra.py \
  experiment="${HYDRA_EXPERIMENT}" \
  model.name_or_path="${MODEL}" \
  experiment.output_dir="${OUTDIR}" \
  experiment.name="robust-grpo-2gpu-${SLURM_JOB_ID:-0}"

# 消融示例（提交前 export 或改上面变量）:
#   export HYDRA_EXPERIMENT=ablation/no_selection_policy
#   export OUTDIR=/共享盘/.../abl_no_sel_${SLURM_JOB_ID}
