#!/bin/bash
# Submit GPU from CLI (this file has NO gpu #SBATCH lines -- cluster rejects --gpus-per-node here):
#   cd .../local_scripts && bash submit_grpo_3gpu_245_ln207.sh
#   OR: yhbatch --gres=gpu:3 run_grpo_3gpu_245.sh
# If Invalid gres: edit GPU string in submit script; check: scontrol show node NODENAME | grep -i gres
# File must be LF only. Prefer: dos2unix run_grpo_3gpu_245.sh ; avoid sed -i on NFS (can truncate).

#SBATCH -J grpo-gpu134
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p ip001_bmw
#SBATCH -t 48:00:00
#SBATCH -o /XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/xuhongtu/Video-R1-main/src/r1-v/outputs/yhbatch-%j.out
#SBATCH -e /XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/xuhongtu/Video-R1-main/src/r1-v/outputs/yhbatch-%j.err
#SBATCH -D /XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/xuhongtu/Video-R1-main/src/r1-v

set -euo pipefail

source /APP/u22/ai_x86/toolshs/set-XY-I.sh 2>/dev/null || true

module purge 2>/dev/null || true
module load CUDA/12.3 2>/dev/null || module load CUDA/12.2 2>/dev/null
export CUDA_HOME=/APP/u22/ai_x86/CUDA/12.2
export PATH="$CUDA_HOME/bin:$PATH"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME

# Checkpoints & logs on XYFS (large pool), not under $HOME
export OUTDIR=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/xuhongtu/Video-R1-main/src/r1-v/outputs
export R1V=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/xuhongtu/Video-R1-main/src/r1-v
export VENV=/HOME/lzu_bmwang/lzu_bmwang_1/.conda/envs/video-r1
export MODEL=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/pretrained/Qwen2.5-VL-7B-COT-SFT
export DATA=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/xuhongtu/Video-R1-main/src/r1-v/Video-R1-data/Video-R1-260k.json
export DS=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/xuhongtu/Video-R1-main/src/r1-v/local_scripts/zero3.json

export TMPDIR=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TORCH_HOME=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/torch_home
export XDG_CACHE_HOME=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/xdg_cache
mkdir -p "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$OUTDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,3,4}

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_QWENVL_VIDEO_READER=decord DECORD_NUM_THREADS=1
export VIDEO_R1_INIT_PROGRESS=1 VIDEO_R1_TRAIN_PROGRESS=1 VIDEO_R1_CUDA_SYNC_LOGS=1 QWEN_VL_VIDEO_DECODE_LOCK_DISABLE=1
export HF_HOME=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export PATH="$VENV/bin:$PATH"
cd "$R1V"

# Fresh run: remove old Trainer checkpoints (logs under OUTDIR are kept).
rm -rf "$OUTDIR"/checkpoint-*

MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-1} % 20000 + 1))
echo "JOB=${SLURM_JOB_ID:-na} MASTER_PORT=$MASTER_PORT RESUME=off CUDA=$CUDA_VISIBLE_DEVICES" >>"$OUTDIR/yhbatch_meta_${SLURM_JOB_ID:-na}.txt"

exec "$VENV/bin/torchrun" --nproc_per_node=3 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  src/open_r1/grpo.py --use_vllm false --deepspeed "$DS" --output_dir "$OUTDIR" --report_to none \
  --model_name_or_path "$MODEL" --dataset_name "$DATA" --bf16 true --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --dataloader_num_workers 6 --learning_rate 1e-6 --lr_scheduler_type cosine \
  --max_prompt_length 768 --max_completion_length 512 --num_generations 4 --max_pixels 24576 --min_pixels 3136 \
  --temporal true --len_control true --beta 0.04 --max_grad_norm 5 --logging_steps 10 --save_steps 60 --save_total_limit 4 \
  --max_steps 300 --gradient_checkpointing true --attn_implementation sdpa
