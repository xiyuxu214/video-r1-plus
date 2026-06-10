# Inference Optimization Script

推理优化脚本，实现多种优化技术以加速推理并降低内存使用。

## 功能特性

### 1. 模型量化
- **INT8量化**: 动态量化，减少模型大小和内存使用
- **FP16/BF16**: 半精度推理，加速计算

### 2. 帧缓存机制
- 避免重复编码相同的视频帧
- LRU缓存策略
- 可配置缓存大小

### 3. 动态批处理
- 根据输入大小动态调整批次
- 自动批处理队列管理
- 支持vLLM自动批处理

### 4. 选择头Early-Exit
- 高置信度时跳过完整推理
- 基于选择策略的快速决策
- 可配置置信度阈值

### 5. 分布式推理支持
- Tensor并行
- Pipeline并行
- 多GPU推理

### 6. 基准测试
- 推理速度测试
- 内存使用监控
- 缓存命中率统计
- Early-exit率统计

## 使用方法

### 基本使用

```bash
# 使用默认配置
python optimize_inference.py --model-path Qwen/Qwen2-VL-7B-Instruct

# 启用量化
python optimize_inference.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --quantization fp16

# 启用缓存和early-exit
python optimize_inference.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --use-cache \
    --cache-size 2000 \
    --early-exit \
    --early-exit-threshold 0.9

# 使用vLLM和分布式推理
python optimize_inference.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --use-vllm \
    --tensor-parallel-size 2 \
    --distributed
```

### 运行基准测试

```bash
# 基本基准测试
python optimize_inference.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --benchmark \
    --test-data test_data.json \
    --num-runs 5

# 完整优化基准测试
python optimize_inference.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --quantization fp16 \
    --use-cache \
    --cache-size 2000 \
    --early-exit \
    --early-exit-threshold 0.95 \
    --use-vllm \
    --benchmark \
    --test-data test_data.json \
    --num-runs 10
```

### 测试数据格式

测试数据应为JSON格式，包含视频路径和问题：

```json
[
    {
        "video_path": "path/to/video1.mp4",
        "question": "What is happening in this video?"
    },
    {
        "video_path": "path/to/video2.mp4",
        "question": "Describe the main action."
    }
]
```

## 配置选项

### 量化选项
- `--quantization int8`: INT8动态量化
- `--quantization fp16`: FP16半精度
- `--quantization bf16`: BF16半精度（需要硬件支持）

### 缓存选项
- `--use-cache`: 启用帧缓存
- `--cache-size`: 缓存大小（默认1000）

### 批处理选项
- `--dynamic-batching`: 启用动态批处理
- `--max-batch-size`: 最大批次大小（默认8）

### Early-Exit选项
- `--early-exit`: 启用early-exit
- `--early-exit-threshold`: 置信度阈值（默认0.95）

### 分布式选项
- `--distributed`: 启用分布式推理
- `--tensor-parallel-size`: Tensor并行大小（默认1）
- `--pipeline-parallel-size`: Pipeline并行大小（默认1）

### vLLM选项
- `--use-vllm`: 使用vLLM（推荐，性能更好）

## 基准测试输出

基准测试会生成以下输出：

1. **控制台输出**: 实时显示测试进度和结果
2. **benchmark_results.json**: 详细的JSON格式结果

### 输出指标

- `total_time_mean`: 平均总时间
- `time_per_sample_mean`: 每个样本平均时间
- `throughput`: 吞吐量（样本/秒）
- `memory_used_mb`: 内存使用（MB）
- `cache_hit_rate`: 缓存命中率
- `early_exit_rate`: Early-exit率

### 示例输出

```
================================================================================
Benchmark Results
================================================================================
Number of samples: 100
Number of runs: 5
Total time: 45.23 ± 2.15 seconds
Time per sample: 452.30 ± 21.50 ms
Throughput: 2.21 samples/second
Memory used: 1234.56 MB

Statistics:
  total_requests: 100
  cache_hits: 85
  early_exits: 12
  avg_inference_time: 0.42
  cache_hit_rate: 0.85
  early_exit_rate: 0.12
================================================================================
```

## 性能优化建议

1. **量化**: 使用FP16可以显著减少内存使用并加速推理
2. **缓存**: 对于重复的视频帧，缓存可以大幅提升速度
3. **vLLM**: 使用vLLM可以获得更好的批处理性能
4. **Early-Exit**: 对于高置信度的情况，可以跳过完整推理
5. **分布式**: 对于大模型，使用多GPU可以提升吞吐量

## 注意事项

1. **量化精度**: INT8量化可能会略微降低精度，建议先测试
2. **缓存内存**: 大缓存会占用更多内存，需要平衡
3. **Early-Exit**: 阈值设置过低可能影响准确性
4. **vLLM依赖**: 需要安装vLLM才能使用相关功能
5. **硬件要求**: BF16需要支持bfloat16的GPU（如A100）

## 故障排除

1. **内存不足**: 尝试使用量化或减小批次大小
2. **vLLM错误**: 确保vLLM版本兼容，或使用transformers后端
3. **缓存问题**: 如果缓存导致问题，可以禁用缓存
4. **分布式错误**: 确保所有GPU可用，检查CUDA版本

