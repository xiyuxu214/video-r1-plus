# LaTeX Table Generator

自动从实验结果生成LaTeX表格的脚本。

## 功能

生成以下类型的LaTeX表格：

1. **主结果表**：与SOTA方法对比
2. **消融研究表**：各组件贡献分析
3. **鲁棒性分析表**：不同噪声条件下的性能
4. **计算效率对比表**：推理速度和内存使用
5. **选择决策分析表**：选择策略的详细分析

## 使用方法

### 基本使用

```bash
# 生成所有表格
python latex_table_generator.py \
    --results-file results.json \
    --output-dir ./latex_tables

# 生成特定类型的表格
python latex_table_generator.py \
    --results-file results.json \
    --table-type ablation \
    --output-dir ./latex_tables

# 指定小数位数
python latex_table_generator.py \
    --results-file results.json \
    --decimal-places 3 \
    --output-dir ./latex_tables
```

## 输入数据格式

输入数据应为JSON格式，包含以下部分：

### 完整示例

```json
{
  "main_results": {
    "vsibench": {
      "accuracy": 0.85,
      "f1_score": 0.82
    },
    "videommmu": {
      "accuracy": 0.78,
      "f1_score": 0.75
    }
  },
  "baselines": {
    "Baseline1": {
      "vsibench": {
        "accuracy": 0.80,
        "f1_score": 0.77
      },
      "videommmu": {
        "accuracy": 0.73,
        "f1_score": 0.70
      }
    },
    "Baseline2": {
      "vsibench": {
        "accuracy": 0.82,
        "f1_score": 0.79
      },
      "videommmu": {
        "accuracy": 0.75,
        "f1_score": 0.72
      }
    }
  },
  "ablation_results": {
    "wo_selection": {
      "accuracy": 0.80,
      "robust_accuracy": 0.72
    },
    "wo_robust_reward": {
      "accuracy": 0.82,
      "robust_accuracy": 0.74
    },
    "wo_temporal": {
      "accuracy": 0.83,
      "robust_accuracy": 0.75
    }
  },
  "baseline_result": {
    "accuracy": 0.85,
    "robust_accuracy": 0.78
  },
  "robustness_results": {
    "vsibench": {
      "base_accuracy": 0.85,
      "robust_accuracy": 0.78,
      "selection_accuracy": 0.82,
      "performance_gap": 0.07,
      "temporal_sensitivity": 0.05
    },
    "videommmu": {
      "base_accuracy": 0.78,
      "robust_accuracy": 0.71,
      "selection_accuracy": 0.76,
      "performance_gap": 0.07,
      "temporal_sensitivity": 0.06
    }
  },
  "efficiency_results": {
    "Ours": {
      "inference_time_ms": 452.3,
      "memory_usage_mb": 1234.56,
      "throughput_samples_per_sec": 2.21,
      "cache_hit_rate": 0.85,
      "early_exit_rate": 0.12
    },
    "Baseline": {
      "inference_time_ms": 523.7,
      "memory_usage_mb": 1456.78,
      "throughput_samples_per_sec": 1.91,
      "cache_hit_rate": 0.0,
      "early_exit_rate": 0.0
    }
  },
  "selection_results": {
    "vsibench": {
      "clean_selection_rate": 0.65,
      "noisy_selection_rate": 0.35,
      "correct_selection_rate": 0.82,
      "avg_selection_confidence": 0.89,
      "early_exit_rate": 0.12
    },
    "videommmu": {
      "clean_selection_rate": 0.68,
      "noisy_selection_rate": 0.32,
      "correct_selection_rate": 0.79,
      "avg_selection_confidence": 0.87,
      "early_exit_rate": 0.10
    }
  }
}
```

### 数据字段说明

#### main_results
主实验结果，格式：`{benchmark: {metric: value}}`

#### baselines
基线方法结果，格式：`{model_name: {benchmark: {metric: value}}}`

#### ablation_results
消融实验结果，格式：`{experiment_name: {metric: value}}`
- 实验名称支持 `wo_` 前缀，会自动转换为 "w/o ..."

#### baseline_result
完整模型的基线结果，用于消融研究对比

#### robustness_results
鲁棒性分析结果，格式：`{benchmark: {metric: value}}`
- `base_accuracy`: 基础准确率（0-1）
- `robust_accuracy`: 鲁棒准确率（0-1）
- `selection_accuracy`: 选择准确率（0-1）
- `performance_gap`: 性能差距（0-1）
- `temporal_sensitivity`: 时序敏感性（0-1）

#### efficiency_results
计算效率结果，格式：`{model: {metric: value}}`
- `inference_time_ms`: 推理时间（毫秒）
- `memory_usage_mb`: 内存使用（MB）
- `throughput_samples_per_sec`: 吞吐量（样本/秒）
- `cache_hit_rate`: 缓存命中率（0-1）
- `early_exit_rate`: Early-exit率（0-1）

#### selection_results
选择决策分析结果，格式：`{benchmark: {metric: value}}`
- `clean_selection_rate`: 选择干净数据的比例（0-1）
- `noisy_selection_rate`: 选择含噪数据的比例（0-1）
- `correct_selection_rate`: 正确选择的比例（0-1）
- `avg_selection_confidence`: 平均选择置信度（0-1）
- `early_exit_rate`: Early-exit率（0-1）

## 输出文件

脚本会在输出目录生成以下文件：

- `main_results_table.tex`: 主结果表
- `ablation_table.tex`: 消融研究表
- `robustness_table.tex`: 鲁棒性分析表
- `efficiency_table.tex`: 计算效率表
- `selection_table.tex`: 选择决策分析表
- `all_tables.tex`: 所有表格的汇总文件

## LaTeX依赖

生成的表格需要以下LaTeX包：

```latex
\usepackage{booktabs}  % 用于 \toprule, \midrule, \bottomrule
\usepackage{multirow}  % 用于多行合并（如果需要）
```

## 表格特性

1. **自动格式化**：
   - 百分比自动转换为 `\%`
   - 特殊字符自动转义
   - 数值自动格式化

2. **高亮显示**：
   - 我们的方法结果自动加粗
   - 平均值行自动加粗

3. **灵活配置**：
   - 可配置小数位数
   - 支持自定义指标名称
   - 支持缺失值处理（显示为 "---"）

## 示例输出

生成的LaTeX表格示例：

```latex
\begin{table}[t]
\centering
\caption{Ablation Study Results}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
Configuration & Accuracy & Robust Accuracy \\
\midrule
Full Model & \textbf{0.85} & \textbf{0.78} \\
\midrule
w/o Selection & 0.80 & 0.72 \\
w/o Robust Reward & 0.82 & 0.74 \\
w/o Temporal & 0.83 & 0.75 \\
\bottomrule
\end{tabular}
\end{table}
```

## 注意事项

1. **数据格式**：确保所有数值为浮点数，不要使用字符串
2. **缺失值**：使用 `null` 或省略字段表示缺失值
3. **百分比**：准确率等指标应使用0-1范围，脚本会自动转换为百分比
4. **特殊字符**：模型名称中的特殊字符会自动转义
5. **表格宽度**：对于宽表格，主结果表使用 `table*` 环境以跨双栏

## 故障排除

1. **JSON解析错误**：检查JSON格式是否正确
2. **缺失字段**：某些表格类型需要特定字段，确保数据完整
3. **LaTeX编译错误**：检查是否安装了必要的LaTeX包
4. **数值格式**：确保所有数值为数字类型，不是字符串

