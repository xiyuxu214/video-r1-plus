# Video-R1 快速开始指南

本指南提供最简化的步骤，帮助您快速运行Video-R1项目。

## 🚀 5分钟快速开始

### 步骤1: 环境准备

```bash
# 创建conda环境
conda create -n video-r1 python=3.11
conda activate video-r1

# 进入项目目录
cd Video-R1/src/r1-v

# 安装项目
pip install -e .
```

### 步骤2: 下载数据

```bash
# 安装git-lfs
git lfs install

# 下载训练数据
git clone https://huggingface.co/datasets/Video-R1/Video-R1-data
cd ../..
python src/unzip.py
```

### 步骤3: 使用预训练模型进行推理

```bash
# 下载模型（或使用自己的模型）
# 模型路径: https://huggingface.co/Video-R1/Video-R1-7B

# 运行推理示例
cd src/r1-v/src/open_r1
python optimize_inference.py \
    --model-path Video-R1/Video-R1-7B \
    --use-vllm
```

## 📝 完整训练流程（简化版）

### 1. SFT训练（可选，可使用提供的模型）

```bash
bash ../../scripts/run_sft_video.sh
```

### 2. RL训练

```bash
cd src/r1-v/src/open_r1

# 使用Hydra配置（最简单）
python train_robust_grpo_hydra.py
```

### 3. 评估

```bash
python evaluate_robustness.py \
    --model_path ../../outputs/robust_grpo \
    --benchmarks vsibench
```

## 🔧 最小配置示例

创建 `config/experiment/quickstart.yaml`:

```yaml
defaults:
  - ../base
  - _self_

experiment:
  name: "quickstart"
  output_dir: "./outputs/quickstart"

model:
  name_or_path: "Qwen/Qwen2-VL-7B-Instruct"

training:
  num_train_epochs: 1
  max_steps: 100  # 快速测试用
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
```

运行：

```bash
python train_robust_grpo_hydra.py experiment=quickstart
```

## ⚡ 常见命令速查

```bash
# 训练
python train_robust_grpo_hydra.py

# 评估
python evaluate_robustness.py --model_path ./outputs/model

# 推理优化
python optimize_inference.py --model-path ./outputs/model

# 消融研究
python ablation_study.py --mode all

# 生成表格
python latex_table_generator.py --results-file results.json

# 生成图表
python method_diagram.py
```

## 🐛 遇到问题？

1. **查看完整文档**: `USAGE_GUIDE.md`
2. **检查常见问题**: 文档中的"常见问题"部分
3. **提交Issue**: GitHub Issues页面

---

**更多详细信息请参考 `USAGE_GUIDE.md`**

