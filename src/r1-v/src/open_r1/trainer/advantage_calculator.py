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
Advantage Calculator Implementation

Complete advantage calculation system with:
1. Group-based advantage calculation (ordered, shuffled, noisy)
2. GAE (Generalized Advantage Estimation)
3. Advantage normalization and importance sampling weights
4. PPO-Clip loss computation

Formula: A_i = (R_i - μ_R) / σ_R + λ * selection_advantage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


class AdvantageCalculator:
    """
    优势计算器
    
    实现多种优势计算方法和PPO-Clip损失。
    
    Args:
        selection_advantage_weight (float): 选择优势权重λ，默认0.1
        gae_lambda (float): GAE的λ参数，默认0.95
        gae_gamma (float): GAE的折扣因子γ，默认0.99
        use_gae (bool): 是否使用GAE，默认False
        normalize_advantages (bool): 是否标准化优势，默认True
        clip_epsilon (float): PPO-Clip的ε参数，默认0.2
        value_loss_coef (float): 价值损失系数，默认0.5
        entropy_coef (float): 熵系数，默认0.01
        max_grad_norm (float): 最大梯度范数，默认1.0
    """
    
    def __init__(
        self,
        selection_advantage_weight: float = 0.1,
        gae_lambda: float = 0.95,
        gae_gamma: float = 0.99,
        use_gae: bool = False,
        normalize_advantages: bool = True,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.selection_advantage_weight = selection_advantage_weight
        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma
        self.use_gae = use_gae
        self.normalize_advantages = normalize_advantages
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        selection_rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算分组优势
        
        按组计算优势：A_i = (R_i - μ_R) / σ_R + λ * selection_advantage
        
        Args:
            rewards: 总奖励，形状为 (batch_size,)
            group_size: 每组的大小（例如，有序组、乱序组、含噪组各有多少样本）
            selection_rewards: 选择奖励，形状为 (batch_size,)，可选
            
        Returns:
            优势张量，形状为 (batch_size,)
        """
        batch_size = rewards.size(0)
        num_groups = batch_size // group_size
        
        # 重塑为 (num_groups, group_size)
        rewards_reshaped = rewards.view(num_groups, group_size)
        
        # 计算每组的均值和标准差
        group_means = rewards_reshaped.mean(dim=1, keepdim=True)  # (num_groups, 1)
        group_stds = rewards_reshaped.std(dim=1, keepdim=True)  # (num_groups, 1)
        group_stds = torch.clamp(group_stds, min=1e-4)  # 避免除零
        
        # 标准化奖励：(R_i - μ_R) / σ_R
        normalized_rewards = (rewards_reshaped - group_means) / group_stds
        
        # 计算选择优势（如果提供）
        selection_advantage = torch.zeros_like(normalized_rewards)
        if selection_rewards is not None:
            selection_rewards_reshaped = selection_rewards.view(num_groups, group_size)
            selection_group_means = selection_rewards_reshaped.mean(dim=1, keepdim=True)
            selection_group_stds = selection_rewards_reshaped.std(dim=1, keepdim=True)
            selection_group_stds = torch.clamp(selection_group_stds, min=1e-4)
            
            # 标准化选择奖励
            selection_advantage = (selection_rewards_reshaped - selection_group_means) / selection_group_stds
        
        # 组合优势：A_i = (R_i - μ_R) / σ_R + λ * selection_advantage
        advantages = normalized_rewards + self.selection_advantage_weight * selection_advantage
        
        # 展平回原始形状
        advantages = advantages.view(batch_size)
        
        return advantages
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE（Generalized Advantage Estimation）
        
        GAE公式：
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
        
        Args:
            rewards: 奖励序列，形状为 (batch_size, sequence_length) 或 (batch_size,)
            values: 价值估计，形状为 (batch_size, sequence_length) 或 (batch_size,)
            next_values: 下一个状态的价值估计，形状与values相同，可选
            dones: 是否结束标志，形状与rewards相同，可选
            
        Returns:
            Tuple[advantages, returns]:
                - advantages: GAE优势，形状与rewards相同
                - returns: 回报（用于价值函数训练），形状与rewards相同
        """
        if rewards.dim() == 1:
            # 单步情况，转换为序列
            rewards = rewards.unsqueeze(1)
            values = values.unsqueeze(1)
            if next_values is not None:
                next_values = next_values.unsqueeze(1)
            if dones is not None:
                dones = dones.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, sequence_length = rewards.shape
        
        # 如果没有提供next_values，使用values的下一时刻值
        if next_values is None:
            # 对于序列，next_values是values向右移动一位
            next_values = torch.zeros_like(values)
            next_values[:, :-1] = values[:, 1:]
            # 最后一时刻的next_value设为0（或使用bootstrapping）
            next_values[:, -1] = values[:, -1]  # 或者设为0，取决于具体场景
        
        # 如果没有提供dones，假设所有步骤都未结束
        if dones is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool)
        
        # 计算TD误差：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        deltas = rewards + self.gae_gamma * next_values * (~dones).float() - values
        
        # 计算GAE优势（从后向前计算）
        advantages = torch.zeros_like(deltas)
        gae = 0
        
        for t in reversed(range(sequence_length)):
            gae = deltas[:, t] + self.gae_gamma * self.gae_lambda * (~dones[:, t]).float() * gae
            advantages[:, t] = gae
        
        # 计算回报：returns = advantages + values
        returns = advantages + values
        
        if squeeze_output:
            advantages = advantages.squeeze(1)
            returns = returns.squeeze(1)
        
        return advantages, returns
    
    def normalize_advantages(
        self,
        advantages: torch.Tensor,
        group_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        标准化优势
        
        Args:
            advantages: 优势张量，形状为 (batch_size,)
            group_size: 分组大小，如果提供则按组标准化，否则全局标准化
            
        Returns:
            标准化后的优势
        """
        if group_size is not None:
            # 按组标准化
            batch_size = advantages.size(0)
            num_groups = batch_size // group_size
            advantages_reshaped = advantages.view(num_groups, group_size)
            
            group_means = advantages_reshaped.mean(dim=1, keepdim=True)
            group_stds = advantages_reshaped.std(dim=1, keepdim=True)
            group_stds = torch.clamp(group_stds, min=1e-4)
            
            normalized = (advantages_reshaped - group_means) / group_stds
            return normalized.view(batch_size)
        else:
            # 全局标准化
            mean = advantages.mean()
            std = advantages.std()
            std = torch.clamp(std, min=1e-4)
            return (advantages - mean) / std
    
    def compute_importance_sampling_weights(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算重要性采样权重
        
        r(θ) = exp(log π_θ(a|s) - log π_old(a|s))
        
        Args:
            old_log_probs: 旧策略的对数概率，形状为 (batch_size, sequence_length) 或 (batch_size,)
            new_log_probs: 新策略的对数概率，形状与old_log_probs相同
            
        Returns:
            重要性采样权重，形状与输入相同
        """
        # 计算重要性采样比率
        log_ratio = new_log_probs - old_log_probs
        
        # 如果是对序列，需要对序列长度维度求和
        if log_ratio.dim() > 1:
            log_ratio = log_ratio.sum(dim=-1)
        
        # 计算权重：r(θ) = exp(log_ratio)
        weights = torch.exp(log_ratio)
        
        # 裁剪权重以避免数值不稳定
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        return weights
    
    def compute_ppo_clip_loss(
        self,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算PPO-Clip损失
        
        L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
        
        Args:
            advantages: 优势，形状为 (batch_size,)
            old_log_probs: 旧策略的对数概率，形状为 (batch_size, sequence_length) 或 (batch_size,)
            new_log_probs: 新策略的对数概率，形状与old_log_probs相同
            values: 价值估计，可选，用于价值损失
            returns: 回报，可选，用于价值损失
            entropy: 策略熵，可选，用于熵奖励
            
        Returns:
            包含以下键的字典：
                - "policy_loss": 策略损失
                - "value_loss": 价值损失（如果提供values和returns）
                - "entropy": 熵（如果提供）
                - "total_loss": 总损失
        """
        # 计算重要性采样比率
        log_ratio = new_log_probs - old_log_probs
        
        # 如果是对序列，需要对序列长度维度求和
        if log_ratio.dim() > 1:
            log_ratio = log_ratio.sum(dim=-1)
        
        ratio = torch.exp(log_ratio)
        
        # 确保advantages和ratio形状匹配
        if advantages.dim() == 0:
            advantages = advantages.unsqueeze(0)
        if ratio.dim() == 0:
            ratio = ratio.unsqueeze(0)
        
        # 扩展advantages以匹配ratio的形状（如果需要）
        if ratio.size(0) != advantages.size(0):
            # 如果ratio是序列，advantages需要扩展
            if ratio.dim() > 1:
                advantages = advantages.unsqueeze(1).expand_as(ratio)
            else:
                # 如果advantages是序列，ratio需要扩展
                advantages = advantages.expand_as(ratio)
        
        # 计算未裁剪的策略损失
        policy_loss_1 = ratio * advantages
        
        # 计算裁剪后的策略损失
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_loss_2 = clipped_ratio * advantages
        
        # PPO-Clip损失：取最小值
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        losses = {
            "policy_loss": policy_loss,
        }
        
        # 计算价值损失（如果提供）
        if values is not None and returns is not None:
            # 确保形状匹配
            if values.dim() > 1:
                values = values.sum(dim=-1)
            if returns.dim() > 1:
                returns = returns.sum(dim=-1)
            
            # 价值损失：MSE
            value_loss = F.mse_loss(values, returns)
            losses["value_loss"] = value_loss * self.value_loss_coef
        else:
            losses["value_loss"] = torch.tensor(0.0, device=advantages.device)
        
        # 计算熵奖励（如果提供）
        if entropy is not None:
            if entropy.dim() > 1:
                entropy = entropy.mean(dim=-1).mean()
            else:
                entropy = entropy.mean()
            losses["entropy"] = entropy
            # 熵作为奖励（负损失）
            entropy_bonus = entropy * self.entropy_coef
        else:
            entropy_bonus = torch.tensor(0.0, device=advantages.device)
            losses["entropy"] = torch.tensor(0.0, device=advantages.device)
        
        # 总损失
        total_loss = policy_loss + losses["value_loss"] - entropy_bonus
        losses["total_loss"] = total_loss
        
        return losses
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        selection_rewards: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        next_values: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        计算优势（统一接口）
        
        根据配置选择使用分组优势或GAE。
        
        Args:
            rewards: 总奖励，形状为 (batch_size,)
            group_size: 每组的大小
            selection_rewards: 选择奖励，可选
            values: 价值估计（用于GAE），可选
            next_values: 下一个状态的价值估计（用于GAE），可选
            dones: 是否结束标志（用于GAE），可选
            normalize: 是否标准化，如果为None则使用self.normalize_advantages
            
        Returns:
            优势张量，形状为 (batch_size,)
        """
        if self.use_gae and values is not None:
            # 使用GAE
            advantages, returns = self.compute_gae(
                rewards, values, next_values, dones
            )
        else:
            # 使用分组优势
            advantages = self.compute_group_advantages(
                rewards, group_size, selection_rewards
            )
        
        # 标准化
        if normalize is None:
            normalize = self.normalize_advantages
        
        if normalize:
            advantages = self.normalize_advantages(advantages, group_size)
        
        return advantages
    
    def compute_kl_divergence(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算KL散度
        
        KL(π_old || π_new) = E[log π_old(a|s) - log π_new(a|s)]
        
        Args:
            old_log_probs: 旧策略的对数概率
            new_log_probs: 新策略的对数概率
            
        Returns:
            KL散度
        """
        log_ratio = old_log_probs - new_log_probs
        
        # 如果是对序列，需要对序列长度维度求和
        if log_ratio.dim() > 1:
            log_ratio = log_ratio.sum(dim=-1)
        
        # KL散度
        kl = log_ratio.mean()
        
        return kl


def create_advantage_calculator(
    selection_advantage_weight: float = 0.1,
    gae_lambda: float = 0.95,
    gae_gamma: float = 0.99,
    use_gae: bool = False,
    normalize_advantages: bool = True,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
) -> AdvantageCalculator:
    """
    创建优势计算器的便捷函数
    
    Args:
        selection_advantage_weight: 选择优势权重λ
        gae_lambda: GAE的λ参数
        gae_gamma: GAE的折扣因子γ
        use_gae: 是否使用GAE
        normalize_advantages: 是否标准化优势
        clip_epsilon: PPO-Clip的ε参数
        value_loss_coef: 价值损失系数
        entropy_coef: 熵系数
        max_grad_norm: 最大梯度范数
        
    Returns:
        AdvantageCalculator实例
    """
    return AdvantageCalculator(
        selection_advantage_weight=selection_advantage_weight,
        gae_lambda=gae_lambda,
        gae_gamma=gae_gamma,
        use_gae=use_gae,
        normalize_advantages=normalize_advantages,
        clip_epsilon=clip_epsilon,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
    )

