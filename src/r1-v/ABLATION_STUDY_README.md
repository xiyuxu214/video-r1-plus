# Ablation Study Script

自动生成消融实验配置、运行实验并汇总结果的脚本。

## 功能

1. **自动生成消融实验配置**
   - 移除选择机制 (wo_selection)
   - 移除鲁棒奖励 (wo_robust_reward)
   - 移除时序对比 (wo_temporal)
   - 移除图像数据 (wo_image)
   - 不同噪声类型对比 (spatial_only, temporal_only, combined_noise)
   - 不同选择头架构对比 (selection_mlp, selection_transformer, selection_lstm)

2. **自动运行实验**
   - 支持单个或批量运行
   - 支持试运行模式（dry-run）
   - 自动收集实验结果

3. **结果汇总**
   - 生成对比表格（CSV、HTML、Markdown）
   - 生成详细报告
   - 支持与基线对比

## 使用方法

### 1. 仅生成配置文件

```bash
python ablation_study.py --mode generate
```

这将在 `./config/experiment/ablation/` 目录下生成所有消融实验配置文件。

### 2. 运行所有消融实验

```bash
# 实际运行
python ablation_study.py --mode all

# 试运行（仅生成配置，不实际执行）
python ablation_study.py --mode all --dry-run
```

### 3. 运行特定实验

```bash
# 运行特定实验
python ablation_study.py --mode run --experiments wo_selection wo_robust_reward

# 试运行
python ablation_study.py --mode run --experiments wo_selection --dry-run
```

### 4. 仅汇总已有结果

```bash
python ablation_study.py --mode aggregate
```

这会从 `./experiments/ablation/` 目录加载所有实验结果并生成对比表格。

## 输出文件

运行完成后，会在 `./experiments/ablation/` 目录下生成：

1. **ablation_results.json**: 所有实验的原始结果
2. **ablation_comparison.csv**: 对比表格（CSV格式）
3. **ablation_comparison.html**: 对比表格（HTML格式）
4. **ablation_comparison.md**: 对比表格（Markdown格式）
5. **ablation_report.md**: 详细报告

## 生成的消融实验

### 1. wo_selection
移除选择机制，禁用选择策略和选择奖励。

### 2. wo_robust_reward
移除鲁棒性奖励，仅使用基础正确性奖励。

### 3. wo_temporal
移除时序对比，禁用有序vs乱序组对比。

### 4. wo_image
移除图像数据，仅使用视频数据。

### 5. spatial_only
仅使用空间噪声（高斯模糊、遮挡、色彩抖动）。

### 6. temporal_only
仅使用时序噪声（帧重复、帧丢弃、顺序错乱）。

### 7. combined_noise
组合所有噪声类型（空间+时序+语义）。

### 8. selection_mlp
使用MLP架构的选择头（默认）。

### 9. selection_transformer
使用Transformer架构的选择头。

### 10. selection_lstm
使用LSTM架构的选择头。

## 自定义配置

可以通过修改 `ablation_study.py` 中的 `generate_ablation_configs` 方法来添加新的消融实验。

## 注意事项

1. **训练时间**: 运行所有消融实验可能需要很长时间，建议先试运行确认配置正确。

2. **资源需求**: 确保有足够的GPU资源，或使用 `--experiments` 参数分批运行。

3. **结果路径**: 确保实验输出目录有写入权限。

4. **依赖**: 需要安装以下包：
   - pandas
   - pyyaml
   - tabulate (用于Markdown表格)

## 示例输出

对比表格示例：

| Experiment | Status | base_accuracy | robust_accuracy | selection_accuracy |
|------------|--------|--------------|-----------------|---------------------|
| wo_selection | completed | 0.85 | 0.78 | N/A |
| wo_robust_reward | completed | 0.87 | 0.75 | 0.82 |
| wo_temporal | completed | 0.86 | 0.79 | 0.81 |
| ... | ... | ... | ... | ... |

## 故障排除

1. **配置文件未找到**: 确保先运行 `--mode generate` 生成配置文件。

2. **实验结果未找到**: 检查实验输出目录，确保训练已完成并生成了 `final_metrics.json`。

3. **导入错误**: 确保所有依赖包已安装，特别是 `pandas` 和 `yaml`。

