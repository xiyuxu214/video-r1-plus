# Method Diagram Generator

使用Graphviz自动生成方法图表的脚本。

## 功能

生成以下类型的图表：

1. **Robust-T-GRPO整体架构图**：展示完整的训练和推理流程
2. **双阶段选择机制流程图**：详细的选择策略流程
3. **奖励计算数据流图**：各种奖励组件的计算流程
4. **训练-推理对比图**：训练和推理阶段的对比

## 安装依赖

### 1. 安装Python包

```bash
pip install graphviz
```

### 2. 安装Graphviz系统包

**Windows:**
- 下载安装包：https://graphviz.org/download/
- 安装后，将安装目录的bin文件夹添加到系统PATH

**Linux:**
```bash
sudo apt-get install graphviz
# 或
sudo yum install graphviz
```

**macOS:**
```bash
brew install graphviz
```

## 使用方法

### 基本使用

```bash
# 生成所有图表（PDF和PNG格式）
python method_diagram.py

# 指定输出目录
python method_diagram.py --output-dir ./my_diagrams

# 只生成PDF格式
python method_diagram.py --format pdf

# 只生成PNG格式
python method_diagram.py --format png

# 只生成特定类型的图表
python method_diagram.py --diagram-type architecture
python method_diagram.py --diagram-type selection
python method_diagram.py --diagram-type reward
python method_diagram.py --diagram-type comparison
```

### 组合使用

```bash
# 只生成架构图的PDF版本
python method_diagram.py \
    --diagram-type architecture \
    --format pdf \
    --output-dir ./diagrams
```

## 生成的图表

### 1. Robust-T-GRPO整体架构图

展示完整的系统架构，包括：
- 输入层（视频和问题）
- 选择策略层（视觉编码器和选择头）
- 三组对比（有序、乱序、含噪）
- 模型推理（视频-语言模型）
- 奖励计算（多种奖励组件）
- 策略优化（优势计算和PPO损失）

**输出文件**: `robust_t_grpo_architecture.pdf` / `.png`

### 2. 双阶段选择机制流程图

详细展示选择策略的两个阶段：
- **阶段1：特征提取**
  - 编码干净帧
  - 编码含噪帧
  - 编码问题文本
- **阶段2：选择决策**
  - 特征拼接
  - MLP层处理
  - 生成选择概率
  - 置信度检查
  - Early-exit决策

**输出文件**: `selection_mechanism_flowchart.pdf` / `.png`

### 3. 奖励计算数据流图

展示奖励计算的完整流程：
- **输入数据**：预测、真实值、性能指标等
- **奖励组件**：正确性、时序、选择、鲁棒性、长度
- **权重配置**：各奖励组件的权重
- **总奖励计算**：加权求和

**输出文件**: `reward_computation_dataflow.pdf` / `.png`

### 4. 训练-推理对比图

对比训练和推理阶段的关键差异：
- **训练阶段**：
  - 多生成（G=8）
  - 奖励计算
  - 优势计算
  - 梯度更新
  - 课程学习
- **推理阶段**：
  - 单生成
  - 选择策略
  - 帧缓存（可选）
  - Early-exit（可选）

**输出文件**: `training_inference_comparison.pdf` / `.png`

## 输出格式

支持以下格式：
- **PDF**: 矢量图，适合论文使用
- **PNG**: 位图，适合演示和网页
- **SVG**: 矢量图，适合网页和编辑

默认同时生成PDF和PNG格式。

## 图表特性

1. **高分辨率**：所有图表使用300 DPI，确保清晰度
2. **颜色编码**：不同模块使用不同颜色，便于区分
3. **层次结构**：使用子图（cluster）组织相关组件
4. **连接标注**：边标签说明数据流向
5. **样式区分**：实线表示数据流，虚线表示控制流

## 自定义

可以通过修改代码来自定义：
- 颜色方案
- 节点形状
- 字体大小
- 图表尺寸
- 布局方向

## 故障排除

### 1. Graphviz未找到

**错误**: `graphviz.backend.ExecutableNotFound`

**解决**:
- 确保已安装Graphviz系统包
- 检查PATH环境变量
- Windows用户需要重启终端

### 2. 权限错误

**错误**: 无法写入输出目录

**解决**:
- 检查输出目录权限
- 使用绝对路径
- 确保有写入权限

### 3. 图表显示不完整

**解决**:
- 调整图表尺寸（修改`size`参数）
- 减小字体大小
- 简化节点标签

### 4. 中文显示问题

**解决**:
- 确保系统安装了中文字体
- 在代码中指定字体：`dot.attr('node', fontname='SimHei')`

## 示例输出

运行脚本后，会在输出目录生成：

```
diagrams/
├── robust_t_grpo_architecture.pdf
├── robust_t_grpo_architecture.png
├── selection_mechanism_flowchart.pdf
├── selection_mechanism_flowchart.png
├── reward_computation_dataflow.pdf
├── reward_computation_dataflow.png
├── training_inference_comparison.pdf
└── training_inference_comparison.png
```

## 在论文中使用

生成的PDF文件可以直接插入LaTeX文档：

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{diagrams/robust_t_grpo_architecture.pdf}
    \caption{Robust-T-GRPO整体架构}
    \label{fig:architecture}
\end{figure}
```

## 注意事项

1. **首次运行**：可能需要较长时间生成所有图表
2. **文件大小**：PNG文件可能较大，PDF文件较小
3. **编辑**：PDF和SVG格式可以用Inkscape等工具编辑
4. **版本兼容**：确保Graphviz版本 >= 2.40

