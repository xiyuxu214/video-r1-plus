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
Visualization Module

Complete visualization toolkit for Robust-T-GRPO training and evaluation:
1. Selection decision visualization
2. Reward components trend analysis
3. Noise robustness curves
4. Temporal attention visualization
5. Comparison tables with baseline models

Uses matplotlib and seaborn with interactive chart support.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive charts will be disabled.")

# 设置matplotlib和seaborn样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_selection_decision(
    clean_rewards: np.ndarray,
    noisy_rewards: np.ndarray,
    selection_probs: np.ndarray,
    selection_confidence: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    interactive: bool = False,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    可视化模型选择干净/含噪的决策过程
    
    Args:
        clean_rewards: 干净数据的奖励，形状为 (n_samples,)
        noisy_rewards: 含噪数据的奖励，形状为 (n_samples,)
        selection_probs: 选择概率，形状为 (n_samples, 2) 或 (n_samples,)
        selection_confidence: 选择置信度，形状为 (n_samples,)，可选
        output_path: 输出路径，如果为None则不保存
        interactive: 是否生成交互式图表（需要plotly）
        figsize: 图表大小
    """
    if interactive and PLOTLY_AVAILABLE:
        _plot_selection_decision_interactive(
            clean_rewards, noisy_rewards, selection_probs, selection_confidence, output_path
        )
    else:
        _plot_selection_decision_static(
            clean_rewards, noisy_rewards, selection_probs, selection_confidence, output_path, figsize
        )


def _plot_selection_decision_static(
    clean_rewards: np.ndarray,
    noisy_rewards: np.ndarray,
    selection_probs: np.ndarray,
    selection_confidence: Optional[np.ndarray],
    output_path: Optional[str],
    figsize: Tuple[int, int],
):
    """静态版本的选择决策可视化"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Selection Decision Visualization', fontsize=16, fontweight='bold')
    
    # 处理选择概率
    if selection_probs.ndim == 2:
        clean_probs = selection_probs[:, 0]
        noisy_probs = selection_probs[:, 1]
    else:
        clean_probs = selection_probs
        noisy_probs = 1.0 - selection_probs
    
    # 1. 散点图：干净奖励 vs 含噪奖励，颜色表示选择
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        clean_rewards, noisy_rewards,
        c=clean_probs, cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', linewidth=0.5
    )
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal Performance')
    ax1.set_xlabel('Clean Reward', fontsize=12)
    ax1.set_ylabel('Noisy Reward', fontsize=12)
    ax1.set_title('Reward Comparison with Selection Probability', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Clean Selection Probability')
    
    # 2. 选择概率分布
    ax2 = axes[0, 1]
    ax2.hist(clean_probs, bins=30, alpha=0.7, label='Clean Prob', color='green')
    ax2.hist(noisy_probs, bins=30, alpha=0.7, label='Noisy Prob', color='red')
    ax2.set_xlabel('Selection Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Selection Probability Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 置信度 vs 奖励差异
    if selection_confidence is not None:
        ax3 = axes[1, 0]
        reward_diff = clean_rewards - noisy_rewards
        ax3.scatter(selection_confidence, reward_diff, alpha=0.6, s=50)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Selection Confidence', fontsize=12)
        ax3.set_ylabel('Reward Difference (Clean - Noisy)', fontsize=12)
        ax3.set_title('Confidence vs Reward Difference', fontsize=12)
        ax3.grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    # 4. 决策矩阵热力图
    ax4 = axes[1, 1]
    # 创建决策矩阵：根据奖励差异和选择概率
    reward_diff = clean_rewards - noisy_rewards
    decision_matrix = np.zeros((20, 20))
    
    for i, (diff, prob) in enumerate(zip(reward_diff, clean_probs)):
        diff_bin = np.clip(int((diff + 1) * 10), 0, 19)
        prob_bin = np.clip(int(prob * 20), 0, 19)
        decision_matrix[prob_bin, diff_bin] += 1
    
    im = ax4.imshow(decision_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax4.set_xlabel('Reward Difference (Clean - Noisy)', fontsize=12)
    ax4.set_ylabel('Clean Selection Probability', fontsize=12)
    ax4.set_title('Decision Matrix Heatmap', fontsize=12)
    plt.colorbar(im, ax=ax4, label='Count')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Selection decision plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def _plot_selection_decision_interactive(
    clean_rewards: np.ndarray,
    noisy_rewards: np.ndarray,
    selection_probs: np.ndarray,
    selection_confidence: Optional[np.ndarray],
    output_path: Optional[str],
):
    """交互式版本的选择决策可视化"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, using static version")
        return
    
    if selection_probs.ndim == 2:
        clean_probs = selection_probs[:, 0]
        noisy_probs = selection_probs[:, 1]
    else:
        clean_probs = selection_probs
        noisy_probs = 1.0 - selection_probs
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Reward Comparison with Selection',
            'Selection Probability Distribution',
            'Confidence vs Reward Difference',
            'Decision Matrix'
        ),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    # 1. 散点图
    fig.add_trace(
        go.Scatter(
            x=clean_rewards, y=noisy_rewards,
            mode='markers',
            marker=dict(
                size=10, color=clean_probs,
                colorscale='RdYlGn', showscale=True,
                colorbar=dict(title="Clean Prob", x=1.15)
            ),
            text=[f"Clean: {c:.2f}, Noisy: {n:.2f}, Prob: {p:.2f}" 
                  for c, n, p in zip(clean_rewards, noisy_rewards, clean_probs)],
            hovertemplate='<b>Clean Reward:</b> %{x:.3f}<br>' +
                         '<b>Noisy Reward:</b> %{y:.3f}<br>' +
                         '<b>Clean Prob:</b> %{marker.color:.3f}<extra></extra>',
            name='Samples'
        ),
        row=1, col=1
    )
    
    # 对角线
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                  line=dict(dash='dash', color='black'),
                  name='Equal Performance', showlegend=False),
        row=1, col=1
    )
    
    # 2. 概率分布
    fig.add_trace(
        go.Histogram(x=clean_probs, name='Clean Prob', opacity=0.7, nbinsx=30),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=noisy_probs, name='Noisy Prob', opacity=0.7, nbinsx=30),
        row=1, col=2
    )
    
    # 3. 置信度 vs 奖励差异
    if selection_confidence is not None:
        reward_diff = clean_rewards - noisy_rewards
        fig.add_trace(
            go.Scatter(
                x=selection_confidence, y=reward_diff,
                mode='markers', name='Samples',
                hovertemplate='<b>Confidence:</b> %{x:.3f}<br>' +
                             '<b>Reward Diff:</b> %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    # 4. 决策矩阵
    reward_diff = clean_rewards - noisy_rewards
    decision_matrix = np.zeros((20, 20))
    for diff, prob in zip(reward_diff, clean_probs):
        diff_bin = np.clip(int((diff + 1) * 10), 0, 19)
        prob_bin = np.clip(int(prob * 20), 0, 19)
        decision_matrix[prob_bin, diff_bin] += 1
    
    fig.add_trace(
        go.Heatmap(z=decision_matrix, colorscale='YlOrRd', showscale=True),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800, title_text="Selection Decision Visualization",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Clean Reward", row=1, col=1)
    fig.update_yaxes(title_text="Noisy Reward", row=1, col=1)
    fig.update_xaxes(title_text="Selection Probability", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    if selection_confidence is not None:
        fig.update_xaxes(title_text="Selection Confidence", row=2, col=1)
        fig.update_yaxes(title_text="Reward Difference", row=2, col=1)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        if output_path.endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_image(output_path)
        print(f"Interactive selection decision plot saved to: {output_path}")
    else:
        fig.show()


def plot_reward_components(
    reward_history: Dict[str, List[float]],
    steps: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    interactive: bool = False,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """
    展示各奖励分量的变化趋势
    
    Args:
        reward_history: 奖励历史字典，键为奖励名称，值为奖励值列表
        steps: 步数列表，如果为None则使用索引
        output_path: 输出路径
        interactive: 是否生成交互式图表
        figsize: 图表大小
    """
    if steps is None:
        steps = list(range(len(list(reward_history.values())[0])))
    
    if interactive and PLOTLY_AVAILABLE:
        _plot_reward_components_interactive(reward_history, steps, output_path)
    else:
        _plot_reward_components_static(reward_history, steps, output_path, figsize)


def _plot_reward_components_static(
    reward_history: Dict[str, List[float]],
    steps: List[int],
    output_path: Optional[str],
    figsize: Tuple[int, int],
):
    """静态版本的奖励分量可视化"""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Reward Components Trend', fontsize=16, fontweight='bold')
    
    # 1. 各奖励分量趋势
    ax1 = axes[0]
    for name, values in reward_history.items():
        ax1.plot(steps, values, label=name, linewidth=2, alpha=0.8)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Reward Value', fontsize=12)
    ax1.set_title('Individual Reward Components', fontsize=12)
    ax1.legend(loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. 总奖励趋势
    ax2 = axes[1]
    total_reward = np.sum([np.array(values) for values in reward_history.values()], axis=0)
    ax2.plot(steps, total_reward, label='Total Reward', linewidth=2, color='black')
    ax2.fill_between(steps, total_reward, alpha=0.3, color='gray')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Total Reward', fontsize=12)
    ax2.set_title('Total Reward Trend', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Reward components plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def _plot_reward_components_interactive(
    reward_history: Dict[str, List[float]],
    steps: List[int],
    output_path: Optional[str],
):
    """交互式版本的奖励分量可视化"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, using static version")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Individual Reward Components', 'Total Reward Trend'),
        vertical_spacing=0.1
    )
    
    # 各奖励分量
    for name, values in reward_history.items():
        fig.add_trace(
            go.Scatter(x=steps, y=values, mode='lines', name=name,
                      hovertemplate=f'<b>{name}</b><br>' +
                                   'Step: %{x}<br>' +
                                   'Value: %{y:.3f}<extra></extra>'),
            row=1, col=1
        )
    
    # 总奖励
    total_reward = np.sum([np.array(values) for values in reward_history.values()], axis=0)
    fig.add_trace(
        go.Scatter(x=steps, y=total_reward, mode='lines', name='Total Reward',
                  line=dict(color='black', width=3),
                  fill='tozeroy', fillcolor='rgba(128,128,128,0.3)',
                  hovertemplate='<b>Total Reward</b><br>' +
                               'Step: %{x}<br>' +
                               'Value: %{y:.3f}<extra></extra>'),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Reward Components Trend", showlegend=True)
    fig.update_xaxes(title_text="Training Steps", row=2, col=1)
    fig.update_yaxes(title_text="Reward Value", row=1, col=1)
    fig.update_yaxes(title_text="Total Reward", row=2, col=1)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        if output_path.endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_image(output_path)
        print(f"Interactive reward components plot saved to: {output_path}")
    else:
        fig.show()


def plot_noise_robustness(
    noise_levels: np.ndarray,
    accuracies: np.ndarray,
    baseline_accuracy: Optional[float] = None,
    output_path: Optional[str] = None,
    interactive: bool = False,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    不同噪声强度下的性能曲线
    
    Args:
        noise_levels: 噪声强度数组
        accuracies: 对应的准确率数组
        baseline_accuracy: 基线准确率（干净数据），可选
        output_path: 输出路径
        interactive: 是否生成交互式图表
        figsize: 图表大小
    """
    if interactive and PLOTLY_AVAILABLE:
        _plot_noise_robustness_interactive(noise_levels, accuracies, baseline_accuracy, output_path)
    else:
        _plot_noise_robustness_static(noise_levels, accuracies, baseline_accuracy, output_path, figsize)


def _plot_noise_robustness_static(
    noise_levels: np.ndarray,
    accuracies: np.ndarray,
    baseline_accuracy: Optional[float],
    output_path: Optional[str],
    figsize: Tuple[int, int],
):
    """静态版本的噪声鲁棒性可视化"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(noise_levels, accuracies, 'o-', linewidth=2, markersize=8, 
            label='Model Performance', color='blue')
    ax.fill_between(noise_levels, accuracies, alpha=0.3, color='blue')
    
    if baseline_accuracy is not None:
        ax.axhline(y=baseline_accuracy, color='green', linestyle='--', 
                  linewidth=2, label=f'Baseline (Clean): {baseline_accuracy:.3f}')
    
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Noise Robustness Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Noise robustness plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def _plot_noise_robustness_interactive(
    noise_levels: np.ndarray,
    accuracies: np.ndarray,
    baseline_accuracy: Optional[float],
    output_path: Optional[str],
):
    """交互式版本的噪声鲁棒性可视化"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, using static version")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=noise_levels, y=accuracies,
        mode='lines+markers',
        name='Model Performance',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        fill='tozeroy', fillcolor='rgba(0,0,255,0.2)',
        hovertemplate='<b>Noise Level:</b> %{x:.3f}<br>' +
                     '<b>Accuracy:</b> %{y:.3f}<extra></extra>'
    ))
    
    if baseline_accuracy is not None:
        fig.add_hline(
            y=baseline_accuracy,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Baseline: {baseline_accuracy:.3f}",
            annotation_position="right"
        )
    
    fig.update_layout(
        title='Noise Robustness Curve',
        xaxis_title='Noise Level',
        yaxis_title='Accuracy',
        hovermode='x unified',
        height=500
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        if output_path.endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_image(output_path)
        print(f"Interactive noise robustness plot saved to: {output_path}")
    else:
        fig.show()


def plot_temporal_attention(
    attention_weights: np.ndarray,
    frame_indices: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    interactive: bool = False,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    可视化模型对时序的关注度
    
    Args:
        attention_weights: 注意力权重，形状为 (n_samples, n_frames) 或 (n_frames,)
        frame_indices: 帧索引，如果为None则使用0到n_frames-1
        output_path: 输出路径
        interactive: 是否生成交互式图表
        figsize: 图表大小
    """
    if attention_weights.ndim == 1:
        attention_weights = attention_weights.reshape(1, -1)
    
    n_samples, n_frames = attention_weights.shape
    
    if frame_indices is None:
        frame_indices = np.arange(n_frames)
    
    if interactive and PLOTLY_AVAILABLE:
        _plot_temporal_attention_interactive(attention_weights, frame_indices, output_path)
    else:
        _plot_temporal_attention_static(attention_weights, frame_indices, output_path, figsize)


def _plot_temporal_attention_static(
    attention_weights: np.ndarray,
    frame_indices: np.ndarray,
    output_path: Optional[str],
    figsize: Tuple[int, int],
):
    """静态版本的时序注意力可视化"""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Temporal Attention Visualization', fontsize=16, fontweight='bold')
    
    # 1. 热力图
    ax1 = axes[0]
    im = ax1.imshow(attention_weights, cmap='YlOrRd', aspect='auto', origin='lower')
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('Sample Index', fontsize=12)
    ax1.set_title('Attention Weights Heatmap', fontsize=12)
    ax1.set_xticks(range(len(frame_indices)))
    ax1.set_xticklabels(frame_indices)
    plt.colorbar(im, ax=ax1, label='Attention Weight')
    
    # 2. 平均注意力
    ax2 = axes[1]
    mean_attention = attention_weights.mean(axis=0)
    std_attention = attention_weights.std(axis=0)
    ax2.plot(frame_indices, mean_attention, 'o-', linewidth=2, markersize=6, label='Mean')
    ax2.fill_between(frame_indices, 
                     mean_attention - std_attention,
                     mean_attention + std_attention,
                     alpha=0.3, label='±1 Std')
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Attention Weight', fontsize=12)
    ax2.set_title('Average Attention Across Samples', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Temporal attention plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def _plot_temporal_attention_interactive(
    attention_weights: np.ndarray,
    frame_indices: np.ndarray,
    output_path: Optional[str],
):
    """交互式版本的时序注意力可视化"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, using static version")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Attention Weights Heatmap', 'Average Attention'),
        vertical_spacing=0.1
    )
    
    # 热力图
    fig.add_trace(
        go.Heatmap(
            z=attention_weights,
            x=frame_indices,
            y=list(range(attention_weights.shape[0])),
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Attention Weight", x=1.15)
        ),
        row=1, col=1
    )
    
    # 平均注意力
    mean_attention = attention_weights.mean(axis=0)
    std_attention = attention_weights.std(axis=0)
    
    fig.add_trace(
        go.Scatter(
            x=frame_indices, y=mean_attention,
            mode='lines+markers', name='Mean',
            line=dict(width=3),
            error_y=dict(type='data', array=std_attention, visible=True),
            hovertemplate='<b>Frame:</b> %{x}<br>' +
                        '<b>Attention:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Temporal Attention Visualization")
    fig.update_xaxes(title_text="Frame Index", row=1, col=1)
    fig.update_yaxes(title_text="Sample Index", row=1, col=1)
    fig.update_xaxes(title_text="Frame Index", row=2, col=1)
    fig.update_yaxes(title_text="Attention Weight", row=2, col=1)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        if output_path.endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_image(output_path)
        print(f"Interactive temporal attention plot saved to: {output_path}")
    else:
        fig.show()


def generate_comparison_table(
    model_results: Dict[str, Dict[str, float]],
    baseline_results: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: Optional[str] = None,
    format: str = 'html',  # 'html', 'latex', 'markdown', 'csv'
) -> str:
    """
    生成与基线模型的对比表格
    
    Args:
        model_results: 模型结果字典，格式为 {benchmark: {metric: value}}
        baseline_results: 基线结果字典，格式相同，可选
        output_path: 输出路径
        format: 输出格式 ('html', 'latex', 'markdown', 'csv')
        
    Returns:
        表格字符串
    """
    # 准备数据
    all_benchmarks = set(model_results.keys())
    if baseline_results:
        all_benchmarks.update(baseline_results.keys())
    
    all_metrics = set()
    for results in [model_results, baseline_results]:
        if results:
            for bench_results in results.values():
                all_metrics.update(bench_results.keys())
    
    # 创建DataFrame
    data = []
    for benchmark in sorted(all_benchmarks):
        row = {'Benchmark': benchmark}
        
        model_bench = model_results.get(benchmark, {})
        baseline_bench = baseline_results.get(benchmark, {}) if baseline_results else {}
        
        for metric in sorted(all_metrics):
            model_val = model_bench.get(metric, None)
            baseline_val = baseline_bench.get(metric, None)
            
            if model_val is not None:
                row[f'{metric} (Model)'] = f'{model_val:.4f}'
            else:
                row[f'{metric} (Model)'] = 'N/A'
            
            if baseline_val is not None:
                row[f'{metric} (Baseline)'] = f'{baseline_val:.4f}'
            else:
                row[f'{metric} (Baseline)'] = 'N/A'
            
            # 计算改进
            if model_val is not None and baseline_val is not None:
                improvement = model_val - baseline_val
                row[f'{metric} (Δ)'] = f'{improvement:+.4f}'
            else:
                row[f'{metric} (Δ)'] = 'N/A'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 生成表格
    if format == 'html':
        table_str = df.to_html(index=False, escape=False, classes='table table-striped')
    elif format == 'latex':
        table_str = df.to_latex(index=False, escape=False)
    elif format == 'markdown':
        table_str = df.to_markdown(index=False)
    elif format == 'csv':
        table_str = df.to_csv(index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # 保存文件
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(table_str)
        print(f"Comparison table saved to: {output_path}")
    
    return table_str


# 便捷函数
def visualize_training_summary(
    training_log: Dict[str, Any],
    output_dir: str = './visualizations',
    interactive: bool = False,
):
    """
    生成训练摘要可视化
    
    Args:
        training_log: 训练日志字典
        output_dir: 输出目录
        interactive: 是否生成交互式图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    steps = training_log.get('steps', [])
    reward_history = training_log.get('rewards', {})
    
    # 生成各种可视化
    if reward_history:
        plot_reward_components(
            reward_history, steps,
            output_path=os.path.join(output_dir, 'reward_components.html' if interactive else 'reward_components.png'),
            interactive=interactive
        )
    
    print(f"Training summary visualizations saved to: {output_dir}")


if __name__ == "__main__":
    # 示例使用
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("  - plot_selection_decision")
    print("  - plot_reward_components")
    print("  - plot_noise_robustness")
    print("  - plot_temporal_attention")
    print("  - generate_comparison_table")

