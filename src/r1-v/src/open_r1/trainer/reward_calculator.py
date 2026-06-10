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
Reward Calculator Implementation

Integrated reward calculation system with multiple reward functions:
1. Correctness reward: Based on answer matching (supports multiple choice, numerical, text)
2. Temporal reward: Based on ordered vs shuffled group performance comparison
3. Selection reward: Encourages selecting clean data
4. Robustness penalty: Penalty for selecting noisy data with incorrect reasoning
5. Length penalty: Controls reasoning chain length in reasonable range

All rewards have configurable weights and support dynamic adjustment.
"""

import re
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. Free-form text rewards will be limited.")


class RewardCalculator:
    """
    奖励计算器
    
    集成多种奖励函数，支持动态权重调整。
    
    Args:
        correctness_weight (float): 正确性奖励权重，默认1.0
        temporal_weight (float): 时序奖励权重，默认0.3
        selection_weight (float): 选择奖励权重，默认0.2
        clean_choice_bonus (float): 选择干净数据的基础奖励，默认0.1
        smart_choice_bonus (float): 明智选择的额外奖励，默认0.15
        robustness_penalty_weight (float): 鲁棒性惩罚权重，默认0.2
        length_penalty_weight (float): 长度惩罚权重，默认0.1
        optimal_length_min (int): 最优推理链长度下限，默认200
        optimal_length_max (int): 最优推理链长度上限，默认512
        length_penalty_strength (float): 长度惩罚强度，默认0.5
    """
    
    def __init__(
        self,
        correctness_weight: float = 1.0,
        temporal_weight: float = 0.3,
        selection_weight: float = 0.2,
        clean_choice_bonus: float = 0.1,
        smart_choice_bonus: float = 0.15,
        robustness_penalty_weight: float = 0.2,
        length_penalty_weight: float = 0.1,
        optimal_length_min: int = 200,
        optimal_length_max: int = 512,
        length_penalty_strength: float = 0.5,
    ):
        # 奖励权重
        self.correctness_weight = correctness_weight
        self.temporal_weight = temporal_weight
        self.selection_weight = selection_weight
        self.robustness_penalty_weight = robustness_penalty_weight
        self.length_penalty_weight = length_penalty_weight
        
        # 选择奖励参数
        self.clean_choice_bonus = clean_choice_bonus
        self.smart_choice_bonus = smart_choice_bonus
        
        # 长度惩罚参数
        self.optimal_length_min = optimal_length_min
        self.optimal_length_max = optimal_length_max
        self.length_penalty_strength = length_penalty_strength
        
        # 初始化ROUGE评分器（如果可用）
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def update_weights(
        self,
        correctness_weight: Optional[float] = None,
        temporal_weight: Optional[float] = None,
        selection_weight: Optional[float] = None,
        clean_choice_bonus: Optional[float] = None,
        smart_choice_bonus: Optional[float] = None,
        robustness_penalty_weight: Optional[float] = None,
        length_penalty_weight: Optional[float] = None,
    ):
        """动态更新奖励权重"""
        if correctness_weight is not None:
            self.correctness_weight = correctness_weight
        if temporal_weight is not None:
            self.temporal_weight = temporal_weight
        if selection_weight is not None:
            self.selection_weight = selection_weight
        if clean_choice_bonus is not None:
            self.clean_choice_bonus = clean_choice_bonus
        if smart_choice_bonus is not None:
            self.smart_choice_bonus = smart_choice_bonus
        if robustness_penalty_weight is not None:
            self.robustness_penalty_weight = robustness_penalty_weight
        if length_penalty_weight is not None:
            self.length_penalty_weight = length_penalty_weight
    
    @staticmethod
    def extract_answer(text: str) -> str:
        """从文本中提取答案"""
        # 支持多种答案格式
        patterns = [
            r'<answer>\s*(.*?)\s*</answer>',
            r'<answer>(.*?)</answer>',
            r'Answer:\s*(.*?)(?:\n|$)',
            r'答案[：:]\s*(.*?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    @staticmethod
    def normalize_number(num_str: str) -> Optional[float]:
        """标准化数字字符串"""
        try:
            num_str = num_str.replace(',', '').replace(' ', '')
            return float(num_str)
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """计算词错误率 (Word Error Rate)"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m, n = len(ref_words), len(hyp_words)
        
        # 动态规划计算编辑距离
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        
        return d[m][n] / max(1, m)
    
    def compute_rouge_score(self, reference: str, hypothesis: str) -> float:
        """计算ROUGE分数"""
        if self.rouge_scorer is None:
            # 如果没有ROUGE，使用简单的重叠度
            ref_words = set(reference.lower().split())
            hyp_words = set(hypothesis.lower().split())
            if len(ref_words) == 0:
                return 0.0
            overlap = len(ref_words & hyp_words)
            return overlap / len(ref_words)
        
        scores = self.rouge_scorer.score(reference, hypothesis)
        average_fmeasure = (
            scores['rouge1'].fmeasure + 
            scores['rouge2'].fmeasure + 
            scores['rougeL'].fmeasure
        ) / 3
        return average_fmeasure
    
    def correctness_reward(
        self,
        completions: List[str],
        solutions: List[str],
        question_types: List[str],
    ) -> torch.Tensor:
        """
        计算正确性奖励
        
        支持多种问题类型：multiple choice, numerical, OCR, free-form, regression
        
        Args:
            completions: 模型生成的完成文本列表
            solutions: 标准答案列表
            question_types: 问题类型列表
            
        Returns:
            正确性奖励张量，形状为 (batch_size,)
        """
        rewards = []
        
        for completion, solution, question_type in zip(completions, solutions, question_types):
            try:
                output_ans = self.extract_answer(completion)
                gt_ans = self.extract_answer(solution)
                
                if question_type == "multiple choice":
                    # 多选题：精确匹配
                    reward = 1.0 if output_ans.strip().lower() == gt_ans.strip().lower() else 0.0
                
                elif question_type == "numerical":
                    # 数值题：考虑小数点和精度
                    gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                    out_has_decimal = ("." in output_ans) or ("," in output_ans)
                    
                    if gt_has_decimal != out_has_decimal:
                        reward = 0.0
                    else:
                        gt_number = self.normalize_number(gt_ans)
                        out_number = self.normalize_number(output_ans)
                        
                        if gt_number is None or out_number is None:
                            reward = 0.0
                        else:
                            # 允许2位小数精度
                            reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
                
                elif question_type == "OCR":
                    # OCR题：使用词错误率
                    error_rate = self.compute_wer(gt_ans, output_ans)
                    reward = max(0.0, min(1.0, 1.0 - error_rate))
                
                elif question_type == "free-form":
                    # 自由文本题：使用ROUGE分数
                    if len(gt_ans) == 0:
                        reward = 0.0
                    else:
                        score = self.compute_rouge_score(gt_ans, output_ans)
                        reward = max(0.0, min(1.0, score))
                
                elif question_type == "regression":
                    # 回归题：使用相对误差
                    gt_number = self.normalize_number(gt_ans)
                    out_number = self.normalize_number(output_ans)
                    
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                        rel_diff = min(1.0, max(0.0, rel_diff))
                        reward = 1.0 - rel_diff
                
                else:
                    # 未知类型，默认0
                    reward = 0.0
            
            except Exception as e:
                print(f"Error computing correctness reward: {e}")
                reward = 0.0
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def temporal_reward(
        self,
        ordered_rewards: torch.Tensor,
        shuffled_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算时序奖励
        
        基于有序组和乱序组的性能对比。如果有序组性能更好，给予奖励。
        
        Args:
            ordered_rewards: 有序组的奖励，形状为 (batch_size,)
            shuffled_rewards: 乱序组的奖励，形状为 (batch_size,)
            
        Returns:
            时序奖励张量，形状为 (batch_size,)
        """
        # 计算性能差异
        performance_diff = ordered_rewards - shuffled_rewards
        
        # 如果有序组更好，给予奖励；如果乱序组更好，给予较小奖励
        # 奖励范围：[0, 1]
        temporal_reward = torch.sigmoid(performance_diff * 5.0)  # 使用sigmoid平滑
        
        return temporal_reward
    
    def selection_reward(
        self,
        selection_probs: torch.Tensor,
        clean_rewards: torch.Tensor,
        noisy_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算选择奖励
        
        包含两部分：
        1. clean_choice_bonus: 选择干净数据的基础奖励
        2. smart_choice_bonus: 当干净版本更好时选择干净的额外奖励
        
        Args:
            selection_probs: 选择概率，形状为 (batch_size, 2)，最后一维为 [clean_prob, noisy_prob]
            clean_rewards: 干净数据的奖励，形状为 (batch_size,)
            noisy_rewards: 含噪数据的奖励，形状为 (batch_size,)
            
        Returns:
            选择奖励张量，形状为 (batch_size,)
        """
        # 提取选择概率
        if selection_probs.dim() == 2:
            clean_prob = selection_probs[:, 0]
            noisy_prob = selection_probs[:, 1]
        else:
            # 如果是一维，假设是clean的概率
            clean_prob = selection_probs
            noisy_prob = 1.0 - selection_probs
        
        # 1. 基础奖励：选择干净数据
        clean_choice_reward = clean_prob * self.clean_choice_bonus
        
        # 2. 明智选择奖励：当干净版本更好时选择干净
        clean_better = clean_rewards > noisy_rewards
        smart_choice = clean_better & (clean_prob > noisy_prob)
        smart_choice_reward = smart_choice.float() * self.smart_choice_bonus
        
        # 总选择奖励
        selection_reward = clean_choice_reward + smart_choice_reward
        
        return selection_reward
    
    def robustness_penalty(
        self,
        selection_probs: torch.Tensor,
        noisy_rewards: torch.Tensor,
        correctness_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算鲁棒性惩罚
        
        当选择含噪数据且推理错误时，给予惩罚。
        
        Args:
            selection_probs: 选择概率，形状为 (batch_size, 2)
            noisy_rewards: 含噪数据的奖励，形状为 (batch_size,)
            correctness_rewards: 正确性奖励，形状为 (batch_size,)
            
        Returns:
            鲁棒性惩罚张量（负值），形状为 (batch_size,)
        """
        # 提取选择含噪数据的概率
        if selection_probs.dim() == 2:
            noisy_prob = selection_probs[:, 1]
        else:
            noisy_prob = 1.0 - selection_probs
        
        # 判断是否推理错误
        incorrect = correctness_rewards < 0.5
        
        # 计算惩罚：选择含噪数据且推理错误
        penalty_mask = noisy_prob > 0.5  # 选择了含噪数据
        penalty_mask = penalty_mask & incorrect  # 且推理错误
        
        # 惩罚强度与含噪数据的选择概率和错误程度相关
        penalty_strength = noisy_prob * (1.0 - correctness_rewards)
        penalty = -penalty_strength * penalty_mask.float()
        
        return penalty
    
    def length_penalty(
        self,
        completion_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算长度惩罚
        
        控制推理链长度在合理范围内。
        
        Args:
            completion_lengths: 完成文本的长度（token数或字符数），形状为 (batch_size,)
            
        Returns:
            长度惩罚张量，形状为 (batch_size,)
        """
        # 将长度转换为tensor（如果还不是）
        if not isinstance(completion_lengths, torch.Tensor):
            completion_lengths = torch.tensor(completion_lengths, dtype=torch.float32)
        
        # 计算长度偏差
        lengths = completion_lengths.float()
        
        # 最优长度范围
        optimal_min = float(self.optimal_length_min)
        optimal_max = float(self.optimal_length_max)
        optimal_center = (optimal_min + optimal_max) / 2.0
        
        # 计算惩罚
        # 如果长度在最优范围内，惩罚为0
        # 如果长度偏离最优范围，惩罚逐渐增加
        penalty = torch.zeros_like(lengths)
        
        # 太短
        too_short = lengths < optimal_min
        penalty[too_short] = (optimal_min - lengths[too_short]) / optimal_min * self.length_penalty_strength
        
        # 太长
        too_long = lengths > optimal_max
        penalty[too_long] = (lengths[too_long] - optimal_max) / optimal_max * self.length_penalty_strength
        
        # 负值表示惩罚
        penalty = -penalty
        
        return penalty
    
    def compute_total_reward(
        self,
        completions: List[str],
        solutions: List[str],
        question_types: List[str],
        ordered_rewards: Optional[torch.Tensor] = None,
        shuffled_rewards: Optional[torch.Tensor] = None,
        selection_probs: Optional[torch.Tensor] = None,
        clean_rewards: Optional[torch.Tensor] = None,
        noisy_rewards: Optional[torch.Tensor] = None,
        completion_lengths: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        计算总奖励
        
        整合所有奖励函数。
        
        Args:
            completions: 模型生成的完成文本列表
            solutions: 标准答案列表
            question_types: 问题类型列表
            ordered_rewards: 有序组的奖励（可选）
            shuffled_rewards: 乱序组的奖励（可选）
            selection_probs: 选择概率（可选）
            clean_rewards: 干净数据的奖励（可选）
            noisy_rewards: 含噪数据的奖励（可选）
            completion_lengths: 完成文本的长度（可选）
            return_components: 是否返回各组件奖励
            
        Returns:
            总奖励张量，如果return_components=True，还返回各组件奖励的字典
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 正确性奖励
        correctness_rewards = self.correctness_reward(completions, solutions, question_types).to(device)
        total_reward = correctness_rewards * self.correctness_weight
        
        components = {
            "correctness": correctness_rewards,
        }
        
        # 2. 时序奖励
        if ordered_rewards is not None and shuffled_rewards is not None:
            temporal_rewards = self.temporal_reward(ordered_rewards, shuffled_rewards)
            total_reward += temporal_rewards * self.temporal_weight
            components["temporal"] = temporal_rewards
        
        # 3. 选择奖励
        if selection_probs is not None and clean_rewards is not None and noisy_rewards is not None:
            selection_rewards = self.selection_reward(selection_probs, clean_rewards, noisy_rewards)
            total_reward += selection_rewards * self.selection_weight
            components["selection"] = selection_rewards
        
        # 4. 鲁棒性惩罚
        if selection_probs is not None and noisy_rewards is not None:
            robustness_penalties = self.robustness_penalty(
                selection_probs, noisy_rewards, correctness_rewards
            )
            total_reward += robustness_penalties * self.robustness_penalty_weight
            components["robustness_penalty"] = robustness_penalties
        
        # 5. 长度惩罚
        if completion_lengths is not None:
            length_penalties = self.length_penalty(completion_lengths)
            total_reward += length_penalties * self.length_penalty_weight
            components["length_penalty"] = length_penalties
        
        if return_components:
            return total_reward, components
        else:
            return total_reward


def create_reward_calculator(
    correctness_weight: float = 1.0,
    temporal_weight: float = 0.3,
    selection_weight: float = 0.2,
    clean_choice_bonus: float = 0.1,
    smart_choice_bonus: float = 0.15,
    robustness_penalty_weight: float = 0.2,
    length_penalty_weight: float = 0.1,
    optimal_length_min: int = 200,
    optimal_length_max: int = 512,
    length_penalty_strength: float = 0.5,
) -> RewardCalculator:
    """
    创建奖励计算器的便捷函数
    
    Args:
        correctness_weight: 正确性奖励权重
        temporal_weight: 时序奖励权重
        selection_weight: 选择奖励权重
        clean_choice_bonus: 选择干净数据的基础奖励
        smart_choice_bonus: 明智选择的额外奖励
        robustness_penalty_weight: 鲁棒性惩罚权重
        length_penalty_weight: 长度惩罚权重
        optimal_length_min: 最优推理链长度下限
        optimal_length_max: 最优推理链长度上限
        length_penalty_strength: 长度惩罚强度
        
    Returns:
        RewardCalculator实例
    """
    return RewardCalculator(
        correctness_weight=correctness_weight,
        temporal_weight=temporal_weight,
        selection_weight=selection_weight,
        clean_choice_bonus=clean_choice_bonus,
        smart_choice_bonus=smart_choice_bonus,
        robustness_penalty_weight=robustness_penalty_weight,
        length_penalty_weight=length_penalty_weight,
        optimal_length_min=optimal_length_min,
        optimal_length_max=optimal_length_max,
        length_penalty_strength=length_penalty_strength,
    )

