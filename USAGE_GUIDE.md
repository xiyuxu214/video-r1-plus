# Video-R1 完整使用指南

本文档提供Video-R1项目的完整使用指南，包括环境安装、训练、评估、推理和工具使用。

## 📋 目录

1. [项目介绍](#项目介绍)
2. [环境要求](#环境要求)
3. [安装步骤](#安装步骤)
4. [项目结构](#项目结构)
5. [配置说明](#配置说明)
6. [训练流程](#训练流程)
7. [评估流程](#评估流程)
8. [推理流程](#推理流程)
9. [工具使用](#工具使用)
10. [常见问题](#常见问题)

---

## 项目介绍

Video-R1是一个基于强化学习的视频推理多模态大语言模型训练框架，主要特点：

- **Robust-T-GRPO**: 鲁棒的时序组相对策略优化方法
- **双阶段选择机制**: 智能选择干净/含噪帧
- **课程学习**: 从低噪声逐渐过渡到高噪声
- **多奖励机制**: 正确性、时序、选择、鲁棒性奖励
- **完整工具链**: 训练、评估、推理、可视化、实验跟踪

---

## 环境要求

### 硬件要求

- **GPU**: 至少4块H100 (96GB) 或 5块A100 (80GB)
- **内存**: 建议64GB以上
- **存储**: 至少500GB可用空间（用于模型和数据集）

### 软件要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+) 或 Windows (WSL2)
- **Python**: 3.10.9 或更高版本
- **CUDA**: 11.8 或更高版本
- **cuDNN**: 8.0 或更高版本

---

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/tulerfeng/Video-R1
cd Video-R1
```

### 2. 创建Conda环境

```bash
conda create -n video-r1 python=3.11
conda activate video-r1
```

### 3. 安装基础依赖

```bash
# 安装项目依赖
cd src/r1-v
pip install -e .

# 安装qwen-vl-utils（支持decord加速）
cd ../qwen-vl-utils
pip install -e .[decord]
cd ../..
```

### 4. 安装特定版本的依赖

由于版本兼容性问题，需要安装特定版本：

```bash
# Transformers（使用提供的版本）
# 下载: https://drive.google.com/file/d/1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF/view?usp=sharing
unzip transformers-main.zip
cd transformers-main
pip install .
cd ..

# vLLM 0.7.2
pip install vllm==0.7.2

# TRL 0.16.0
pip install trl==0.16.0

# DeepSpeed
pip install deepspeed==0.15.4
```

### 5. 安装可选依赖

```bash
# 实验跟踪
pip install wandb mlflow

# 超参数搜索
pip install optuna

# 可视化
pip install matplotlib seaborn plotly

# LaTeX表格生成
pip install pandas tabulate

# 方法图生成
pip install graphviz
# 还需要安装系统包: sudo apt-get install graphviz (Linux)
```

### 6. 下载数据集

```bash
# 安装git-lfs
git lfs install

# 下载训练数据集
git clone https://huggingface.co/datasets/Video-R1/Video-R1-data
mv Video-R1-data src/r1-v/

# 解压数据
python src/unzip.py
```

### 7. 下载评估数据

```bash
# 下载评估数据集
git clone https://huggingface.co/datasets/Video-R1/Video-R1-eval
mv Video-R1-eval/* src/r1-v/Evaluation/
```

---

## 项目结构

```
Video-R1/
├── src/
│   ├── r1-v/                          # 主项目目录
│   │   ├── src/
│   │   │   └── open_r1/              # 核心代码
│   │   │       ├── train_robust_grpo.py      # Robust-T-GRPO训练
│   │   │       ├── train_robust_grpo_hydra.py # Hydra配置训练
│   │   │       ├── evaluate_robustness.py     # 鲁棒性评估
│   │   │       ├── optimize_inference.py      # 推理优化
│   │   │       ├── ablation_study.py          # 消融研究
│   │   │       ├── latex_table_generator.py   # LaTeX表格生成
│   │   │       ├── method_diagram.py          # 方法图生成
│   │   │       └── trainer/                   # 训练器模块
│   │   │           ├── robust_t_grpo_trainer.py
│   │   │           ├── selection_augmented_policy.py
│   │   │           ├── video_noise_augmentation.py
│   │   │           ├── reward_calculator.py
│   │   │           ├── advantage_calculator.py
│   │   │           ├── experiment_tracker.py
│   │   │           └── visualization.py
│   │   ├── config/                    # Hydra配置文件
│   │   │   ├── model/
│   │   │   ├── training/
│   │   │   ├── data/
│   │   │   └── experiment/
│   │   ├── Evaluation/                # 评估数据
│   │   └── Video-R1-data/             # 训练数据
│   ├── scripts/                       # 训练脚本
│   └── qwen-vl-utils/                # Qwen视频处理工具
├── setup.py                          # 项目安装配置
└── README.md                         # 项目说明
```

---

## 配置说明

### Hydra配置系统

项目使用Hydra进行配置管理，配置文件位于 `src/r1-v/config/`：

- **model/**: 模型架构配置
- **training/**: 训练参数配置
- **data/**: 数据和增强配置
- **experiment/**: 实验配置

### 主要配置参数

#### 模型配置 (`config/model/robust_grpo.yaml`)

```yaml
model:
  name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"

vision:
  max_pixels: 12845056
  min_pixels: 3136

generation:
  num_generations: 8
  max_completion_length: 768
```

#### 训练配置 (`config/training/rl_training.yaml`)

```yaml
training:
  learning_rate: 1.0e-6
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  beta: 0.04
  num_train_epochs: 1
```

#### 奖励权重配置 (`config/training/reward_weights.yaml`)

```yaml
rewards:
  correctness_weight: 1.0
  temporal_weight: 0.3
  selection_weight: 0.2
  robustness_penalty_weight: 0.2
```

---

## 训练流程

### 1. 监督微调 (SFT)

首先在Video-R1-COT-165k数据集上进行SFT：

```bash
bash src/scripts/run_sft_video.sh
```

或使用提供的SFT模型：
- [Qwen2.5-VL-7B-COT-SFT](https://huggingface.co/Video-R1/Qwen2.5-VL-7B-COT-SFT)

### 2. 强化学习训练 (RL)

#### 方式1: 使用Hydra配置（推荐）

```bash
cd src/r1-v/src/open_r1

# 使用默认配置
python train_robust_grpo_hydra.py

# 使用特定实验配置
python train_robust_grpo_hydra.py experiment=ablation/no_selection_policy

# 覆盖参数
python train_robust_grpo_hydra.py \
    experiment=base \
    model.name_or_path=Qwen/Qwen2-VL-7B-Instruct \
    training.learning_rate=2e-6
```

#### 方式2: 使用命令行参数

```bash
# 使用DeepSpeed
deepspeed train_robust_grpo.py \
    --deepspeed local_scripts/zero3.json \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name ./Video-R1-data/Video-R1-260k.json \
    --output_dir ./outputs/robust_grpo \
    --curriculum_learning \
    --curriculum_max_noise 0.5 \
    --use_selection_policy \
    --base_reward_weight 1.0 \
    --selection_reward_weight 0.2 \
    --robustness_reward_weight 0.3 \
    --bf16 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --num_train_epochs 1

# 使用Accelerate
accelerate launch --config_file configs/ddp.yaml train_robust_grpo.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name ./Video-R1-data/Video-R1-260k.json \
    --output_dir ./outputs/robust_grpo \
    --curriculum_learning \
    --use_selection_policy
```

#### 方式3: 使用vLLM加速

```bash
bash src/scripts/run_grpo_vllm_qwen25vl.sh
```

### 3. 训练监控

训练过程中可以使用WandB监控：

```bash
# 在配置中启用WandB
# config/experiment/base.yaml
experiment:
  tracker:
    use_wandb: true
```

---

## 评估流程

### 1. 鲁棒性评估

```bash
cd src/r1-v/src/open_r1

python evaluate_robustness.py \
    --model_path ./outputs/robust_grpo \
    --benchmarks vsibench videommmu mvbench \
    --output_dir ./eval_results \
    --use_selection_policy \
    --target_num_frames 32
```

### 2. 标准基准测试评估

```bash
# 使用提供的评估脚本
bash src/eval_bench.sh
```

### 3. 单样本推理测试

```bash
python src/inference_example.py
```

---

## 推理流程

### 1. 基础推理

```python
from open_r1 import generate

result = generate(
    video_path="path/to/video.mp4",
    question="What is happening in this video?",
    model_path="./outputs/robust_grpo"
)
print(result)
```

### 2. 优化推理

使用推理优化工具：

```bash
python optimize_inference.py \
    --model-path ./outputs/robust_grpo \
    --quantization fp16 \
    --use-cache \
    --cache-size 2000 \
    --early-exit \
    --early-exit-threshold 0.95 \
    --use-vllm \
    --benchmark \
    --test-data test_data.json
```

---

## 工具使用

### 1. 消融研究

自动生成和运行消融实验：

```bash
# 生成所有消融实验配置
python ablation_study.py --mode generate

# 运行所有消融实验
python ablation_study.py --mode all

# 运行特定实验
python ablation_study.py --mode run --experiments wo_selection wo_robust_reward

# 仅汇总已有结果
python ablation_study.py --mode aggregate
```

### 2. 实验跟踪

使用实验跟踪器记录实验：

```python
from trainer import create_experiment_tracker

tracker = create_experiment_tracker(
    experiment_name="robust-grpo-v1",
    project_name="video-understanding",
    use_wandb=True,
    random_seed=42,
)

# 记录指标
tracker.log_metrics({
    "train/accuracy": 0.85,
    "eval/accuracy": 0.82,
}, step=100)

# 结束实验
tracker.finish()
```

### 3. LaTeX表格生成

从实验结果生成LaTeX表格：

```bash
python latex_table_generator.py \
    --results-file results.json \
    --output-dir ./latex_tables \
    --table-type all
```

结果文件格式参考 `LATEX_TABLE_GENERATOR_README.md`。

### 4. 方法图生成

生成方法图表：

```bash
# 生成所有图表（PDF和PNG）
python method_diagram.py

# 生成特定图表
python method_diagram.py --diagram-type architecture --format pdf

# 自定义输出目录
python method_diagram.py --output-dir ./diagrams
```

### 5. 可视化

使用可视化工具：

```python
from trainer.visualization import (
    plot_selection_decision,
    plot_reward_components,
    plot_noise_robustness,
    generate_comparison_table,
)

# 选择决策可视化
plot_selection_decision(
    clean_rewards=clean_rewards,
    noisy_rewards=noisy_rewards,
    selection_probs=selection_probs,
    output_path='./visualizations/selection.html',
    interactive=True,
)
```

---

## 常见问题

### 1. CUDA内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用 `gradient_checkpointing`
- 使用DeepSpeed ZeRO优化
- 减小 `max_pixels` 和 `num_generations`

### 2. Transformers版本冲突

**问题**: 版本不兼容错误

**解决方案**:
- 使用提供的transformers版本
- 不要升级transformers到最新版本

### 3. vLLM安装问题

**问题**: vLLM安装失败

**解决方案**:
- 确保CUDA版本 >= 11.8
- 使用vLLM 0.7.2版本
- 参考vLLM官方安装指南

### 4. 数据集路径错误

**问题**: 找不到数据集文件

**解决方案**:
- 确保数据集在 `src/r1-v/Video-R1-data/` 目录
- 检查JSON文件是否正确解压
- 使用绝对路径

### 5. Graphviz图表生成失败

**问题**: `graphviz.backend.ExecutableNotFound`

**解决方案**:
- 安装Graphviz系统包
- Linux: `sudo apt-get install graphviz`
- Windows: 从官网下载并添加到PATH
- macOS: `brew install graphviz`

### 6. WandB连接问题

**问题**: WandB无法连接

**解决方案**:
- 检查网络连接
- 运行 `wandb login`
- 或禁用WandB: `use_wandb: false`

### 7. 训练速度慢

**问题**: 训练速度过慢

**解决方案**:
- 使用vLLM加速
- 启用混合精度训练 (bf16/fp16)
- 使用多GPU训练
- 优化数据加载 (`num_workers`, `pin_memory`)

### 8. 评估结果不一致

**问题**: 评估结果与论文不一致

**解决方案**:
- 检查评估配置（帧数、分辨率）
- 确保使用正确的解码参数 (top_p=0.001, temperature=0.01)
- 检查评估数据是否正确下载和放置

---

## 快速开始示例

### 完整训练流程

```bash
# 1. 环境设置
conda create -n video-r1 python=3.11
conda activate video-r1
cd src/r1-v
pip install -e .

# 2. 下载数据
git lfs install
git clone https://huggingface.co/datasets/Video-R1/Video-R1-data
python ../../unzip.py

# 3. SFT训练（或使用提供的模型）
bash ../../scripts/run_sft_video.sh

# 4. RL训练
cd src/open_r1
python train_robust_grpo_hydra.py experiment=base

# 5. 评估
python evaluate_robustness.py \
    --model_path ../../outputs/robust_grpo \
    --benchmarks vsibench
```

### 快速推理

```bash
# 使用优化推理
python optimize_inference.py \
    --model-path ./outputs/robust_grpo \
    --quantization fp16 \
    --use-cache \
    --use-vllm
```

---

## 获取帮助

- **GitHub Issues**: [项目Issues页面](https://github.com/tulerfeng/Video-R1/issues)
- **论文**: [arXiv:2503.21776](https://arxiv.org/pdf/2503.21776)
- **模型**: [HuggingFace](https://huggingface.co/Video-R1)

---

## 引用

如果使用本项目，请引用：

```bibtex
@article{feng2025video,
  title={Video-R1: Reinforcing Video Reasoning in MLLMs},
  author={Feng, Kaituo and Gong, Kaixiong and Li, Bohao and Guo, Zonghao and Wang, Yibing and Peng, Tianshuo and Wang, Benyou and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2503.21776},
  year={2025}
}
```

---

## 更新日志

- **2025-03-28**: 初始版本发布
- **2025-09-19**: NeurIPS 2025接收

---

**祝使用愉快！如有问题，请查看常见问题部分或提交Issue。**

