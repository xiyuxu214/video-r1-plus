# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Method Diagram Generator

Generate method diagrams using Graphviz:
1. Robust-T-GRPO overall architecture diagram
2. Two-stage selection mechanism flowchart
3. Reward computation data flow diagram
4. Training-inference comparison diagram

Exports to PDF and PNG formats.
"""

import os
import argparse
from typing import Optional, Dict, List, Tuple
from pathlib import Path

try:
    from graphviz import Digraph, Graph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not available. Please install: pip install graphviz")
    print("Also install Graphviz system package: https://graphviz.org/download/")


class MethodDiagramGenerator:
    """
    方法图生成器
    
    使用Graphviz生成各种方法图表。
    """
    
    def __init__(self, output_dir: str = "./diagrams", format: str = "pdf"):
        self.output_dir = output_dir
        self.format = format
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("graphviz is required. Install with: pip install graphviz")
    
    def generate_architecture_diagram(self) -> str:
        """
        生成Robust-T-GRPO整体架构图
        
        Returns:
            输出文件路径
        """
        dot = Digraph(comment='Robust-T-GRPO Architecture', format=self.format)
        dot.attr(rankdir='TB', size='12,16', dpi=300)
        dot.attr('node', shape='box', style='rounded,filled', fontsize='12')
        dot.attr('edge', fontsize='10')
        
        # 输入层
        with dot.subgraph(name='cluster_input') as input_cluster:
            input_cluster.attr(label='Input', style='filled', color='lightblue')
            input_cluster.node('video', 'Video Input\n(Clean/Noisy)', fillcolor='lightcyan')
            input_cluster.node('question', 'Question Text', fillcolor='lightcyan')
        
        # 选择策略层
        with dot.subgraph(name='cluster_selection') as selection_cluster:
            selection_cluster.attr(label='Selection Policy', style='filled', color='lightgreen')
            selection_cluster.node('vision_encoder', 'Vision Encoder\n(CLIP-ViT)', fillcolor='lightgreen')
            selection_cluster.node('selection_head', 'Selection Head\n(MLP)', fillcolor='lightgreen')
            selection_cluster.node('selection_prob', 'Selection\nProbabilities', fillcolor='lightyellow')
        
        # 三组对比
        with dot.subgraph(name='cluster_groups') as groups_cluster:
            groups_cluster.attr(label='Three Comparison Groups', style='filled', color='lightcoral')
            groups_cluster.node('ordered_group', 'Ordered Group\n(Original Frames)', fillcolor='lightcoral')
            groups_cluster.node('shuffled_group', 'Shuffled Group\n(Random Order)', fillcolor='lightcoral')
            groups_cluster.node('noisy_group', 'Noisy Group\n(With Augmentation)', fillcolor='lightcoral')
        
        # 模型推理
        with dot.subgraph(name='cluster_model') as model_cluster:
            model_cluster.attr(label='Video-Language Model', style='filled', color='lightpink')
            model_cluster.node('vl_model', 'Qwen2-VL\nModel', fillcolor='lightpink')
            model_cluster.node('generations', 'Multiple\nGenerations', fillcolor='lightyellow')
        
        # 奖励计算
        with dot.subgraph(name='cluster_reward') as reward_cluster:
            reward_cluster.attr(label='Reward Computation', style='filled', color='lightgoldenrodyellow')
            reward_cluster.node('correctness', 'Correctness\nReward', fillcolor='lightyellow')
            reward_cluster.node('temporal', 'Temporal\nReward', fillcolor='lightyellow')
            reward_cluster.node('selection', 'Selection\nReward', fillcolor='lightyellow')
            reward_cluster.node('robustness', 'Robustness\nPenalty', fillcolor='lightyellow')
            reward_cluster.node('total_reward', 'Total Reward', fillcolor='gold')
        
        # 策略优化
        with dot.subgraph(name='cluster_optimization') as opt_cluster:
            opt_cluster.attr(label='Policy Optimization', style='filled', color='lightsteelblue')
            opt_cluster.node('advantage', 'Advantage\nCalculation', fillcolor='lightblue')
            opt_cluster.node('ppo_loss', 'PPO-Clip\nLoss', fillcolor='lightblue')
            opt_cluster.node('policy_update', 'Policy Update', fillcolor='lightblue')
        
        # 连接
        dot.edge('video', 'vision_encoder', label='Frames')
        dot.edge('question', 'selection_head', label='Text')
        dot.edge('vision_encoder', 'selection_head', label='Features')
        dot.edge('selection_head', 'selection_prob', label='')
        
        dot.edge('video', 'ordered_group', label='Clean')
        dot.edge('video', 'shuffled_group', label='Shuffled')
        dot.edge('video', 'noisy_group', label='Augmented')
        
        dot.edge('ordered_group', 'vl_model', label='')
        dot.edge('shuffled_group', 'vl_model', label='')
        dot.edge('noisy_group', 'vl_model', label='')
        dot.edge('selection_prob', 'vl_model', label='Selection')
        
        dot.edge('vl_model', 'generations', label='')
        dot.edge('generations', 'correctness', label='Predictions')
        dot.edge('ordered_group', 'temporal', label='Ordered')
        dot.edge('shuffled_group', 'temporal', label='Shuffled')
        dot.edge('selection_prob', 'selection', label='')
        dot.edge('noisy_group', 'robustness', label='Noisy')
        
        dot.edge('correctness', 'total_reward', label='')
        dot.edge('temporal', 'total_reward', label='')
        dot.edge('selection', 'total_reward', label='')
        dot.edge('robustness', 'total_reward', label='')
        
        dot.edge('total_reward', 'advantage', label='')
        dot.edge('advantage', 'ppo_loss', label='')
        dot.edge('ppo_loss', 'policy_update', label='')
        dot.edge('policy_update', 'vl_model', label='Update', style='dashed')
        dot.edge('policy_update', 'selection_head', label='Update', style='dashed')
        
        # 保存
        output_path = os.path.join(self.output_dir, 'robust_t_grpo_architecture')
        dot.render(output_path, cleanup=True)
        print(f"Generated architecture diagram: {output_path}.{self.format}")
        return output_path
    
    def generate_selection_flowchart(self) -> str:
        """
        生成双阶段选择机制流程图
        
        Returns:
            输出文件路径
        """
        dot = Digraph(comment='Two-Stage Selection Mechanism', format=self.format)
        dot.attr(rankdir='TB', size='10,14', dpi=300)
        dot.attr('node', shape='box', style='rounded,filled', fontsize='11')
        dot.attr('edge', fontsize='10')
        
        # 开始
        dot.node('start', 'Start', shape='ellipse', fillcolor='lightgreen')
        
        # 输入
        dot.node('input_clean', 'Clean Frames', fillcolor='lightcyan')
        dot.node('input_noisy', 'Noisy Frames', fillcolor='lightcyan')
        dot.node('input_question', 'Question', fillcolor='lightcyan')
        
        # 第一阶段：特征提取
        with dot.subgraph(name='cluster_stage1') as stage1:
            stage1.attr(label='Stage 1: Feature Extraction', style='filled', color='lightblue')
            stage1.node('encode_clean', 'Encode Clean\nFrames', fillcolor='lightblue')
            stage1.node('encode_noisy', 'Encode Noisy\nFrames', fillcolor='lightblue')
            stage1.node('encode_question', 'Encode Question', fillcolor='lightblue')
            stage1.node('vision_features', 'Vision Features', fillcolor='lightyellow')
            stage1.node('text_features', 'Text Features', fillcolor='lightyellow')
        
        # 第二阶段：选择决策
        with dot.subgraph(name='cluster_stage2') as stage2:
            stage2.attr(label='Stage 2: Selection Decision', style='filled', color='lightgreen')
            stage2.node('concat_features', 'Concatenate\nFeatures', fillcolor='lightgreen')
            stage2.node('mlp_layer1', 'MLP Layer 1\n(512 dim)', fillcolor='lightgreen')
            stage2.node('mlp_layer2', 'MLP Layer 2\n(256 dim)', fillcolor='lightgreen')
            stage2.node('selection_probs', 'Selection\nProbabilities', fillcolor='gold')
            stage2.node('confidence', 'Confidence\nScore', fillcolor='gold')
        
        # 决策分支
        dot.node('check_confidence', 'Confidence\n≥ Threshold?', shape='diamond', fillcolor='lightyellow')
        dot.node('early_exit', 'Early Exit\n(Skip Full Inference)', fillcolor='orange', shape='ellipse')
        dot.node('full_inference', 'Full Inference', fillcolor='lightpink')
        
        # 选择结果
        dot.node('select_clean', 'Select Clean\nFrames', fillcolor='lightgreen', shape='ellipse')
        dot.node('select_noisy', 'Select Noisy\nFrames', fillcolor='lightcoral', shape='ellipse')
        dot.node('end', 'End', shape='ellipse', fillcolor='lightgreen')
        
        # 连接
        dot.edge('start', 'input_clean', label='')
        dot.edge('start', 'input_noisy', label='')
        dot.edge('start', 'input_question', label='')
        
        dot.edge('input_clean', 'encode_clean', label='')
        dot.edge('input_noisy', 'encode_noisy', label='')
        dot.edge('input_question', 'encode_question', label='')
        
        dot.edge('encode_clean', 'vision_features', label='')
        dot.edge('encode_noisy', 'vision_features', label='')
        dot.edge('encode_question', 'text_features', label='')
        
        dot.edge('vision_features', 'concat_features', label='')
        dot.edge('text_features', 'concat_features', label='')
        
        dot.edge('concat_features', 'mlp_layer1', label='')
        dot.edge('mlp_layer1', 'mlp_layer2', label='')
        dot.edge('mlp_layer2', 'selection_probs', label='')
        dot.edge('selection_probs', 'confidence', label='')
        
        dot.edge('confidence', 'check_confidence', label='')
        dot.edge('check_confidence', 'early_exit', label='Yes', style='bold')
        dot.edge('check_confidence', 'full_inference', label='No')
        
        dot.edge('early_exit', 'select_clean', label='High Conf.', style='dashed')
        dot.edge('early_exit', 'select_noisy', label='High Conf.', style='dashed')
        dot.edge('full_inference', 'select_clean', label='')
        dot.edge('full_inference', 'select_noisy', label='')
        
        dot.edge('select_clean', 'end', label='')
        dot.edge('select_noisy', 'end', label='')
        
        # 保存
        output_path = os.path.join(self.output_dir, 'selection_mechanism_flowchart')
        dot.render(output_path, cleanup=True)
        print(f"Generated selection flowchart: {output_path}.{self.format}")
        return output_path
    
    def generate_reward_dataflow(self) -> str:
        """
        生成奖励计算数据流图
        
        Returns:
            输出文件路径
        """
        dot = Digraph(comment='Reward Computation Data Flow', format=self.format)
        dot.attr(rankdir='LR', size='14,10', dpi=300)
        dot.attr('node', shape='box', style='rounded,filled', fontsize='11')
        dot.attr('edge', fontsize='10')
        
        # 输入数据
        with dot.subgraph(name='cluster_inputs') as inputs:
            inputs.attr(label='Input Data', style='filled', color='lightblue')
            inputs.node('predictions', 'Model\nPredictions', fillcolor='lightcyan')
            inputs.node('ground_truth', 'Ground\nTruth', fillcolor='lightcyan')
            inputs.node('ordered_perf', 'Ordered Group\nPerformance', fillcolor='lightcyan')
            inputs.node('shuffled_perf', 'Shuffled Group\nPerformance', fillcolor='lightcyan')
            inputs.node('selection_probs', 'Selection\nProbabilities', fillcolor='lightcyan')
            inputs.node('noisy_perf', 'Noisy Group\nPerformance', fillcolor='lightcyan')
            inputs.node('chain_length', 'Reasoning Chain\nLength', fillcolor='lightcyan')
        
        # 奖励计算组件
        with dot.subgraph(name='cluster_rewards') as rewards:
            rewards.attr(label='Reward Components', style='filled', color='lightgreen')
            rewards.node('correctness', 'Correctness\nReward\n(Accuracy/Format)', fillcolor='lightgreen')
            rewards.node('temporal', 'Temporal\nReward\n(Ordered vs Shuffled)', fillcolor='lightgreen')
            rewards.node('selection', 'Selection\nReward\n(Clean Choice Bonus)', fillcolor='lightgreen')
            rewards.node('robustness', 'Robustness\nPenalty\n(Noisy Performance)', fillcolor='lightcoral')
            rewards.node('length', 'Length\nPenalty\n(Optimal Range)', fillcolor='lightyellow')
        
        # 权重
        with dot.subgraph(name='cluster_weights') as weights:
            weights.attr(label='Reward Weights', style='filled', color='lightgoldenrodyellow')
            weights.node('w_correct', 'w_correct\n= 1.0', fillcolor='gold')
            weights.node('w_temporal', 'w_temporal\n= 0.3', fillcolor='gold')
            weights.node('w_selection', 'w_selection\n= 0.2', fillcolor='gold')
            weights.node('w_robust', 'w_robust\n= 0.2', fillcolor='gold')
            weights.node('w_length', 'w_length\n= 0.1', fillcolor='gold')
        
        # 总奖励
        dot.node('total_reward', 'Total Reward\nR_total', fillcolor='gold', shape='ellipse', style='bold')
        
        # 连接输入到奖励组件
        dot.edge('predictions', 'correctness', label='')
        dot.edge('ground_truth', 'correctness', label='')
        
        dot.edge('ordered_perf', 'temporal', label='')
        dot.edge('shuffled_perf', 'temporal', label='')
        
        dot.edge('selection_probs', 'selection', label='')
        
        dot.edge('noisy_perf', 'robustness', label='')
        
        dot.edge('chain_length', 'length', label='')
        
        # 连接权重
        dot.edge('w_correct', 'correctness', label='×', style='dashed')
        dot.edge('w_temporal', 'temporal', label='×', style='dashed')
        dot.edge('w_selection', 'selection', label='×', style='dashed')
        dot.edge('w_robust', 'robustness', label='×', style='dashed')
        dot.edge('w_length', 'length', label='×', style='dashed')
        
        # 连接到总奖励
        dot.edge('correctness', 'total_reward', label='+')
        dot.edge('temporal', 'total_reward', label='+')
        dot.edge('selection', 'total_reward', label='+')
        dot.edge('robustness', 'total_reward', label='-', style='dashed')
        dot.edge('length', 'total_reward', label='-', style='dashed')
        
        # 保存
        output_path = os.path.join(self.output_dir, 'reward_computation_dataflow')
        dot.render(output_path, cleanup=True)
        print(f"Generated reward dataflow: {output_path}.{self.format}")
        return output_path
    
    def generate_training_inference_comparison(self) -> str:
        """
        生成训练-推理对比图
        
        Returns:
            输出文件路径
        """
        dot = Digraph(comment='Training vs Inference Comparison', format=self.format)
        dot.attr(rankdir='LR', size='16,10', dpi=300)
        dot.attr('node', shape='box', style='rounded,filled', fontsize='11')
        dot.attr('edge', fontsize='10')
        
        # 训练阶段
        with dot.subgraph(name='cluster_training') as training:
            training.attr(label='Training Phase', style='filled', color='lightblue')
            
            training.node('train_input', 'Training Data\n(Clean + Noisy)', fillcolor='lightcyan')
            training.node('train_model', 'Model\nForward Pass', fillcolor='lightblue')
            training.node('train_generations', 'Multiple\nGenerations\n(G=8)', fillcolor='lightyellow')
            training.node('train_reward', 'Reward\nComputation', fillcolor='lightgreen')
            training.node('train_advantage', 'Advantage\nCalculation', fillcolor='lightgreen')
            training.node('train_loss', 'PPO-Clip\nLoss', fillcolor='lightcoral')
            training.node('train_update', 'Gradient\nUpdate', fillcolor='lightcoral')
            training.node('train_curriculum', 'Curriculum\nLearning\n(Noise Schedule)', fillcolor='lightgoldenrodyellow')
        
        # 推理阶段
        with dot.subgraph(name='cluster_inference') as inference:
            inference.attr(label='Inference Phase', style='filled', color='lightgreen')
            
            inference.node('inf_input', 'Test Data\n(Clean)', fillcolor='lightcyan')
            inference.node('inf_selection', 'Selection\nPolicy', fillcolor='lightgreen')
            inference.node('inf_model', 'Model\nForward Pass', fillcolor='lightgreen')
            inference.node('inf_generation', 'Single\nGeneration', fillcolor='lightyellow')
            inference.node('inf_output', 'Final\nPrediction', fillcolor='gold')
            inference.node('inf_cache', 'Frame Cache\n(Optional)', fillcolor='lightblue')
            inference.node('inf_early_exit', 'Early Exit\n(Optional)', fillcolor='orange')
        
        # 训练流程连接
        dot.edge('train_input', 'train_model', label='')
        dot.edge('train_model', 'train_generations', label='')
        dot.edge('train_generations', 'train_reward', label='')
        dot.edge('train_reward', 'train_advantage', label='')
        dot.edge('train_advantage', 'train_loss', label='')
        dot.edge('train_loss', 'train_update', label='')
        dot.edge('train_update', 'train_model', label='Update', style='dashed')
        dot.edge('train_curriculum', 'train_input', label='Adjust', style='dashed')
        
        # 推理流程连接
        dot.edge('inf_input', 'inf_selection', label='')
        dot.edge('inf_selection', 'inf_early_exit', label='High Conf.', style='dashed')
        dot.edge('inf_selection', 'inf_model', label='')
        dot.edge('inf_cache', 'inf_model', label='Cached', style='dashed')
        dot.edge('inf_model', 'inf_generation', label='')
        dot.edge('inf_early_exit', 'inf_output', label='Skip', style='dashed')
        dot.edge('inf_generation', 'inf_output', label='')
        
        # 对比标注
        dot.node('comparison', 'Key Differences:\n• Training: Multiple generations, reward computation, gradient update\n• Inference: Single generation, optional caching and early-exit', 
                shape='note', fillcolor='lightyellow', fontsize='10')
        
        # 保存
        output_path = os.path.join(self.output_dir, 'training_inference_comparison')
        dot.render(output_path, cleanup=True)
        print(f"Generated training-inference comparison: {output_path}.{self.format}")
        return output_path
    
    def generate_all_diagrams(self, formats: List[str] = ['pdf', 'png']) -> Dict[str, List[str]]:
        """
        生成所有图表
        
        Args:
            formats: 输出格式列表
        
        Returns:
            字典，键为图表类型，值为输出文件路径列表
        """
        results = {
            'architecture': [],
            'selection': [],
            'reward': [],
            'comparison': [],
        }
        
        for fmt in formats:
            self.format = fmt
            print(f"\nGenerating diagrams in {fmt.upper()} format...")
            
            results['architecture'].append(self.generate_architecture_diagram())
            results['selection'].append(self.generate_selection_flowchart())
            results['reward'].append(self.generate_reward_dataflow())
            results['comparison'].append(self.generate_training_inference_comparison())
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Method Diagram Generator")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./diagrams',
        help='Output directory for diagrams'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['pdf', 'png', 'svg', 'all'],
        default='all',
        help='Output format (default: all)'
    )
    parser.add_argument(
        '--diagram-type',
        type=str,
        choices=['architecture', 'selection', 'reward', 'comparison', 'all'],
        default='all',
        help='Type of diagram to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    if not GRAPHVIZ_AVAILABLE:
        print("Error: graphviz is not available.")
        print("Please install: pip install graphviz")
        print("Also install Graphviz system package: https://graphviz.org/download/")
        return
    
    generator = MethodDiagramGenerator(output_dir=args.output_dir)
    
    # 确定格式
    if args.format == 'all':
        formats = ['pdf', 'png']
    else:
        formats = [args.format]
    
    # 生成图表
    if args.diagram_type == 'all':
        generator.generate_all_diagrams(formats=formats)
    else:
        for fmt in formats:
            generator.format = fmt
            if args.diagram_type == 'architecture':
                generator.generate_architecture_diagram()
            elif args.diagram_type == 'selection':
                generator.generate_selection_flowchart()
            elif args.diagram_type == 'reward':
                generator.generate_reward_dataflow()
            elif args.diagram_type == 'comparison':
                generator.generate_training_inference_comparison()
    
    print(f"\nAll diagrams generated in: {args.output_dir}")


if __name__ == "__main__":
    main()

