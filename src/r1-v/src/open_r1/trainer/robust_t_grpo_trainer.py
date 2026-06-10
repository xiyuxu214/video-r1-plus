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
Robust-T-GRPO Trainer Implementation

This module implements a robust variant of Temporal Group Relative Policy Optimization (T-GRPO)
that includes three comparison groups: ordered, shuffled, and noisy groups, with adaptive noise scheduling.
"""

import os
import copy
import math
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Dict, List, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class AdaptiveNoiseScheduler:
    """
    自适应噪声调度器
    
    根据训练进度和模型性能动态调整噪声强度。
    
    Args:
        initial_noise_scale (float): 初始噪声缩放因子，默认0.1
        max_noise_scale (float): 最大噪声缩放因子，默认0.5
        min_noise_scale (float): 最小噪声缩放因子，默认0.01
        noise_schedule_type (str): 噪声调度类型，可选 'linear', 'cosine', 'exponential', 'adaptive'
        warmup_steps (int): 预热步数，默认1000
        total_steps (int): 总训练步数
        performance_threshold (float): 性能阈值，用于自适应调度
    """
    
    def __init__(
        self,
        initial_noise_scale: float = 0.1,
        max_noise_scale: float = 0.5,
        min_noise_scale: float = 0.01,
        noise_schedule_type: str = "adaptive",
        warmup_steps: int = 1000,
        total_steps: Optional[int] = None,
        performance_threshold: float = 0.8,
    ):
        self.initial_noise_scale = initial_noise_scale
        self.max_noise_scale = max_noise_scale
        self.min_noise_scale = min_noise_scale
        self.noise_schedule_type = noise_schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.performance_threshold = performance_threshold
        
        self.current_step = 0
        self.recent_rewards: List[float] = []
        self.recent_reward_window = 100  # 用于计算最近平均奖励的窗口大小
        
    def get_noise_scale(self, current_reward: Optional[float] = None) -> float:
        """
        获取当前步的噪声缩放因子
        
        Args:
            current_reward (float, optional): 当前奖励值，用于自适应调度
            
        Returns:
            float: 噪声缩放因子
        """
        self.current_step += 1
        
        if self.noise_schedule_type == "linear":
            if self.total_steps is None:
                return self.initial_noise_scale
            progress = min(1.0, self.current_step / self.total_steps)
            noise_scale = self.initial_noise_scale + (self.max_noise_scale - self.initial_noise_scale) * progress
            
        elif self.noise_schedule_type == "cosine":
            if self.total_steps is None:
                return self.initial_noise_scale
            progress = min(1.0, self.current_step / self.total_steps)
            noise_scale = self.min_noise_scale + (self.max_noise_scale - self.min_noise_scale) * \
                         (1 - math.cos(math.pi * progress)) / 2
            
        elif self.noise_schedule_type == "exponential":
            if self.total_steps is None:
                return self.initial_noise_scale
            progress = min(1.0, self.current_step / self.total_steps)
            decay_rate = math.log(self.min_noise_scale / self.max_noise_scale) / self.total_steps
            noise_scale = self.max_noise_scale * math.exp(decay_rate * self.current_step)
            
        elif self.noise_schedule_type == "adaptive":
            # 自适应调度：根据模型性能调整噪声
            if current_reward is not None:
                self.recent_rewards.append(current_reward)
                if len(self.recent_rewards) > self.recent_reward_window:
                    self.recent_rewards.pop(0)
                
                if len(self.recent_rewards) >= 10:  # 至少需要10个样本
                    avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                    
                    # 如果性能低于阈值，增加噪声以增强鲁棒性
                    if avg_reward < self.performance_threshold:
                        noise_scale = min(
                            self.max_noise_scale,
                            self.initial_noise_scale * (1 + (self.performance_threshold - avg_reward) * 2)
                        )
                    else:
                        # 性能好时逐渐减少噪声
                        noise_scale = max(
                            self.min_noise_scale,
                            self.initial_noise_scale * (avg_reward / self.performance_threshold)
                        )
                else:
                    # 预热阶段：线性增加
                    progress = min(1.0, self.current_step / self.warmup_steps)
                    noise_scale = self.initial_noise_scale * progress
            else:
                # 没有奖励信息时使用线性调度
                if self.current_step < self.warmup_steps:
                    progress = min(1.0, self.current_step / self.warmup_steps)
                    noise_scale = self.initial_noise_scale * progress
                else:
                    noise_scale = self.initial_noise_scale
        else:
            noise_scale = self.initial_noise_scale
            
        # 确保噪声在合理范围内
        noise_scale = max(self.min_noise_scale, min(self.max_noise_scale, noise_scale))
        return noise_scale
    
    def reset(self):
        """重置调度器状态"""
        self.current_step = 0
        self.recent_rewards.clear()


class RobustTGRPOTrainer(transformers.Trainer):
    """
    Robust Temporal Group Relative Policy Optimization (Robust-T-GRPO) Trainer
    
    这是T-GRPO的鲁棒变体，包含三组对比：
    1. 有序组 (Ordered Group): 原始顺序的视频帧
    2. 乱序组 (Shuffled Group): 随机打乱的视频帧
    3. 含噪组 (Noisy Group): 添加噪声的视频帧
    
    该类继承自transformers.Trainer，并实现了鲁棒奖励计算和自适应噪声调度。
    
    Args:
        model (Union[str, PreTrainedModel]): 要训练的模型
        reward_funcs (Union[RewardFunc, list[RewardFunc]]): 奖励函数列表
        args (GRPOConfig, optional): 训练配置
        script_args: 脚本参数
        train_dataset (Optional[Union[Dataset, IterableDataset]]): 训练数据集
        eval_dataset (Optional[Union[Dataset, IterableDataset]]): 评估数据集
        processing_class (Optional[PreTrainedTokenizerBase]): 处理类（tokenizer/processor）
        reward_processing_classes: 奖励函数的处理类
        callbacks (Optional[list[TrainerCallback]]): 回调函数列表
        optimizers: 优化器和调度器
        peft_config: PEFT配置
        max_pixels (Optional[int]): 最大像素数
        min_pixels (Optional[int]): 最小像素数
        attn_implementation (str): 注意力实现方式
        noise_scheduler_config (Optional[Dict]): 噪声调度器配置
        robust_reward_weight (float): 鲁棒奖励权重，默认0.3
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        noise_scheduler_config: Optional[Dict] = None,
        robust_reward_weight: float = 0.3,
    ):
        # 初始化基础配置
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-Robust-T-GRPO")
        
        # 模型初始化
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    f"Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        
        # 参考模型
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None
        
        # 处理类
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
        
        # 奖励函数
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        
        # 奖励处理类
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")
        
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        
        # 数据整理器
        def data_collator(features):
            return features
        
        # 训练参数
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temporal = script_args.temporal if script_args else True
        self.len_control = script_args.len_control if script_args else True
        self.beta = args.beta
        self.robust_reward_weight = robust_reward_weight
        
        # 生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        
        # 乱序组生成配置（一半数量）
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        # 含噪组生成配置（一半数量）
        self.noisy_num_generations = self.num_generations // 2
        self.noisy_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=self.noisy_num_generations,
            pad_token_id=pad_token_id,
        )
        
        # 自适应噪声调度器
        scheduler_config = noise_scheduler_config or {}
        self.noise_scheduler = AdaptiveNoiseScheduler(
            initial_noise_scale=scheduler_config.get("initial_noise_scale", 0.1),
            max_noise_scale=scheduler_config.get("max_noise_scale", 0.5),
            min_noise_scale=scheduler_config.get("min_noise_scale", 0.01),
            noise_schedule_type=scheduler_config.get("noise_schedule_type", "adaptive"),
            warmup_steps=scheduler_config.get("warmup_steps", 1000),
            total_steps=args.max_steps if hasattr(args, "max_steps") else None,
            performance_threshold=scheduler_config.get("performance_threshold", 0.8),
        )
        
        # 禁用警告
        model.warnings_issued["estimate_tokens"] = True
        
        # 初始化指标
        self._metrics = defaultdict(list)
        
        # 调用父类初始化
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        # 梯度累积需要缩放损失
        self.model_accepts_loss_kwargs = False
        
        # 准备参考模型
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
        # 准备奖励模型
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
    
    def _set_signature_columns_if_needed(self):
        """设置签名列"""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    def _get_per_token_logps(self, model, input_ids, **kwargs) -> torch.Tensor:
        """
        获取每个token的对数概率
        
        Args:
            model: 模型
            input_ids: 输入token IDs
            **kwargs: 其他模型输入参数
            
        Returns:
            torch.Tensor: 每个token的对数概率，形状为 (B, L-1)
        """
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V)
        input_ids = input_ids[:, 1:]  # (B, L-1)
        
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        """从数据中移除None值"""
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """准备输入，跳过默认的预处理"""
        return inputs
    
    def _add_noise_to_video(self, video_tensor: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """
        向视频张量添加噪声
        
        Args:
            video_tensor (torch.Tensor): 视频张量，形状为 (T, C, H, W) 或 (B, T, C, H, W)
            noise_scale (float): 噪声缩放因子
            
        Returns:
            torch.Tensor: 添加噪声后的视频张量
        """
        if video_tensor is None or len(video_tensor) == 0:
            return video_tensor
        
        # 确保是4D或5D张量
        if video_tensor.dim() == 4:
            # (T, C, H, W)
            noise = torch.randn_like(video_tensor) * noise_scale
            noisy_video = video_tensor + noise
        elif video_tensor.dim() == 5:
            # (B, T, C, H, W)
            noise = torch.randn_like(video_tensor) * noise_scale
            noisy_video = video_tensor + noise
        else:
            # 其他维度，直接返回
            return video_tensor
        
        # 将值限制在合理范围内（假设输入是归一化的）
        noisy_video = torch.clamp(noisy_video, -1.0, 1.0)
        return noisy_video
    
    def compute_robust_reward(
        self,
        ordered_rewards: torch.Tensor,
        shuffled_rewards: torch.Tensor,
        noisy_rewards: torch.Tensor,
        ordered_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算鲁棒奖励
        
        鲁棒奖励综合考虑三组（有序、乱序、含噪）的表现，鼓励模型在噪声和乱序情况下仍能保持良好性能。
        
        Args:
            ordered_rewards (torch.Tensor): 有序组的奖励，形状为 (B*G,)
            shuffled_rewards (torch.Tensor): 乱序组的奖励，形状为 (B*G_shuffled,)
            noisy_rewards (torch.Tensor): 含噪组的奖励，形状为 (B*G_noisy,)
            ordered_mask (Optional[torch.Tensor]): 有序组的掩码，用于处理不同长度的序列
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 
                - 鲁棒奖励张量，形状与ordered_rewards相同
                - 包含各种统计信息的字典
        """
        device = ordered_rewards.device
        batch_size = ordered_rewards.size(0) // self.num_generations
        
        # 重塑为 (batch_size, num_generations)
        ordered_rewards_reshaped = ordered_rewards.view(batch_size, self.num_generations)
        shuffled_rewards_reshaped = shuffled_rewards.view(batch_size, self.shuffled_num_generations)
        noisy_rewards_reshaped = noisy_rewards.view(batch_size, self.noisy_num_generations)
        
        # 计算各组平均奖励
        ordered_mean = ordered_rewards_reshaped.mean(dim=1)  # (batch_size,)
        shuffled_mean = shuffled_rewards_reshaped.mean(dim=1)  # (batch_size,)
        noisy_mean = noisy_rewards_reshaped.mean(dim=1)  # (batch_size,)
        
        # 计算鲁棒性指标
        # 1. 有序组与乱序组的性能差异（越小越好，说明对顺序不敏感）
        order_robustness = 1.0 - torch.abs(ordered_mean - shuffled_mean) / (ordered_mean + 1e-8)
        order_robustness = torch.clamp(order_robustness, 0.0, 1.0)
        
        # 2. 有序组与含噪组的性能差异（越小越好，说明对噪声不敏感）
        noise_robustness = 1.0 - torch.abs(ordered_mean - noisy_mean) / (ordered_mean + 1e-8)
        noise_robustness = torch.clamp(noise_robustness, 0.0, 1.0)
        
        # 3. 综合鲁棒性分数
        overall_robustness = (order_robustness + noise_robustness) / 2.0
        
        # 4. 计算鲁棒奖励：基础奖励 + 鲁棒性奖励
        # 对于每个有序组的样本，添加基于鲁棒性的奖励
        robust_bonus = overall_robustness * self.robust_reward_weight
        
        # 扩展鲁棒奖励以匹配有序组的数量
        robust_bonus_expanded = robust_bonus.unsqueeze(1).repeat(1, self.num_generations).view(-1)
        
        # 计算最终鲁棒奖励
        robust_rewards = ordered_rewards + robust_bonus_expanded
        
        # 收集统计信息
        stats = {
            "ordered_reward_mean": ordered_rewards.mean().item(),
            "shuffled_reward_mean": shuffled_rewards.mean().item(),
            "noisy_reward_mean": noisy_rewards.mean().item(),
            "order_robustness": order_robustness.mean().item(),
            "noise_robustness": noise_robustness.mean().item(),
            "overall_robustness": overall_robustness.mean().item(),
            "robust_bonus_mean": robust_bonus.mean().item(),
        }
        
        return robust_rewards, stats
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        计算损失函数
        
        实现三组对比（有序、乱序、含噪）的训练流程。
        
        Args:
            model: 要训练的模型
            inputs: 输入数据字典
            return_outputs: 是否返回输出
            num_items_in_batch: 批次中的项目数
            
        Returns:
            torch.Tensor: 损失值
        """
        if return_outputs:
            raise ValueError("The RobustTGRPOTrainer does not support returning outputs")
        
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # 准备输入
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        input_copy = self.remove_none_from_data(input_copy)
        
        if inputs[0]['data_type'] == 'image':
            input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:]
        elif inputs[0]['data_type'] == 'video':
            input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:]
        
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        
        # 处理有序组（原始顺序）
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        _truncate_ids = self.max_prompt_length is not None and inputs[0].get("data_type") not in (
            "image",
            "video",
        )
        if _truncate_ids:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # 处理乱序组（如果启用temporal且有视频）
        shuffled_prompt_inputs = None
        shuffled_prompt_ids = None
        shuffled_prompt_mask = None
        if self.temporal and video_inputs and len(video_inputs) > 0:
            indices = torch.randperm(video_inputs[0].size(0))
            shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
            if _truncate_ids:
                shuffled_prompt_inputs["input_ids"] = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_inputs["attention_mask"] = shuffled_prompt_mask[:, -self.max_prompt_length :]
                shuffled_prompt_ids = shuffled_prompt_inputs["input_ids"]
                shuffled_prompt_mask = shuffled_prompt_inputs["attention_mask"]
        
        # 处理含噪组（添加噪声的视频）
        noisy_prompt_inputs = None
        noisy_prompt_ids = None
        noisy_prompt_mask = None
        if self.temporal and video_inputs and len(video_inputs) > 0:
            # 获取当前噪声缩放因子
            current_noise_scale = self.noise_scheduler.get_noise_scale()
            
            # 添加噪声到视频
            noisy_video_inputs = [self._add_noise_to_video(video_inputs[0], current_noise_scale)]
            
            noisy_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=noisy_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            noisy_prompt_inputs = super()._prepare_inputs(noisy_prompt_inputs)
            noisy_prompt_ids, noisy_prompt_mask = noisy_prompt_inputs["input_ids"], noisy_prompt_inputs["attention_mask"]
            if _truncate_ids:
                noisy_prompt_inputs["input_ids"] = noisy_prompt_ids[:, -self.max_prompt_length :]
                noisy_prompt_inputs["attention_mask"] = noisy_prompt_mask[:, -self.max_prompt_length :]
                noisy_prompt_ids = noisy_prompt_inputs["input_ids"]
                noisy_prompt_mask = noisy_prompt_inputs["attention_mask"]
        
        device = self.accelerator.device
        
        # 生成有序组completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
            # 生成乱序组completions
            shuffled_completion_ids = None
            if shuffled_prompt_inputs is not None:
                shuffled_prompt_completion_ids = unwrapped_model.generate(
                    **shuffled_prompt_inputs,
                    generation_config=self.shuffled_generation_config
                )
                shuffled_prompt_length = shuffled_prompt_ids.size(1)
                shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
            
            # 生成含噪组completions
            noisy_completion_ids = None
            if noisy_prompt_inputs is not None:
                noisy_prompt_completion_ids = unwrapped_model.generate(
                    **noisy_prompt_inputs,
                    generation_config=self.noisy_generation_config
                )
                noisy_prompt_length = noisy_prompt_ids.size(1)
                noisy_prompt_ids = noisy_prompt_completion_ids[:, :noisy_prompt_length]
                noisy_completion_ids = noisy_prompt_completion_ids[:, noisy_prompt_length:]
        
        # 处理EOS token掩码
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # 计算per-token log probabilities
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        
        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        
        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
        
        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1:]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Using fallback.")
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
            per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        # 计算参考模型的per-token log probabilities
        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Using fallback.")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        # 计算KL散度
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        # 解码completions并计算奖励
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        prompts_for_reward = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts_for_reward), len(self.reward_funcs), device=device)
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts_for_reward, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        ordered_rewards = rewards_per_func.sum(dim=1)
        
        # 计算乱序组奖励
        shuffled_rewards = None
        if shuffled_completion_ids is not None:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": c}] for c in shuffled_completions]
            
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(
                    prompts=shuffled_prompts,
                    completions=shuffled_completions,
                    **shuffled_reward_kwargs
                )
                shuffled_rewards_per_func[:, i] = torch.tensor(
                    shuffled_output_reward_func, dtype=torch.float32, device=device
                )
            
            shuffled_rewards = shuffled_rewards_per_func.sum(dim=1)
        
        # 计算含噪组奖励
        noisy_rewards = None
        if noisy_completion_ids is not None:
            noisy_completions = self.processing_class.batch_decode(noisy_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                noisy_completions = [[{"role": "assistant", "content": c}] for c in noisy_completions]
            
            noisy_prompts = [prompt for prompt in prompts for _ in range(self.noisy_num_generations)]
            noisy_rewards_per_func = torch.zeros(len(noisy_prompts), len(self.reward_funcs), device=device)
            
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                noisy_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in noisy_reward_kwargs:
                    for example in inputs:
                        noisy_reward_kwargs[key].extend([example[key]] * self.noisy_num_generations)
                noisy_output_reward_func = reward_func(
                    prompts=noisy_prompts,
                    completions=noisy_completions,
                    **noisy_reward_kwargs
                )
                noisy_rewards_per_func[:, i] = torch.tensor(
                    noisy_output_reward_func, dtype=torch.float32, device=device
                )
            
            noisy_rewards = noisy_rewards_per_func.sum(dim=1)
        
        # 计算鲁棒奖励
        if shuffled_rewards is not None and noisy_rewards is not None:
            robust_rewards, robust_stats = self.compute_robust_reward(
                ordered_rewards, shuffled_rewards, noisy_rewards
            )
            rewards = robust_rewards
        else:
            rewards = ordered_rewards
            robust_stats = {}
        
        # 长度控制（如果启用）
        if self.len_control:
            mask = rewards_per_func[:, 0] > 0.1
            length_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
            if len(selected_indices) > 1:
                for idx in selected_indices:
                    if 320 <= length_list[idx] <= 512:
                        rewards[idx] += 0.2
        
        # 计算分组奖励和优势
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # 计算损失
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 记录指标
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        # 记录鲁棒性指标
        if robust_stats:
            for key, value in robust_stats.items():
                self._metrics[f"robust/{key}"].append(value)
        
        # 记录噪声调度器状态
        self._metrics["noise_scale"].append(self.noise_scheduler.get_noise_scale())
        
        return loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """记录日志"""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """创建模型卡片"""
        if not self.is_world_process_zero():
            return
        
        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None
        
        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]
        tags.append("robust-t-grpo")
        
        citation = textwrap.dedent(
            """\
            @article{robust_t_grpo,
                title        = {{Robust Temporal Group Relative Policy Optimization}},
                author       = {Your Name},
                year         = 2025,
            """
        )
        
        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Robust-T-GRPO",
            trainer_citation=citation,
            paper_title="Robust Temporal Group Relative Policy Optimization",
        )
        
        model_card.save(os.path.join(self.args.output_dir, "README.md"))

