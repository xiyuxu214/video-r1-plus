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
Video Noise Augmentation Pipeline

Complete video data processing pipeline with:
- Frame sampling: uniform, keyframe, random
- Noise augmentation layers:
  - Spatial noise: Gaussian blur, occlusion, color jitter
  - Temporal noise: frame repetition, frame dropping, order shuffling
  - Semantic noise: adding irrelevant frames, partial occlusion of key objects
- Batch assembly: each sample contains clean, shuffled, and noisy versions
- Augmentation probability scheduling: adjust noise intensity by training epoch
"""

import os
import random
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_video
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Some augmentation features will be disabled.")


class FrameSamplingStrategy(Enum):
    """帧采样策略"""
    UNIFORM = "uniform"  # 均匀采样
    KEYFRAME = "keyframe"  # 关键帧采样
    RANDOM = "random"  # 随机采样


class AugmentationProbabilityScheduler:
    """
    数据增强概率调度器
    
    根据训练轮次调整噪声增强的概率和强度。
    """
    
    def __init__(
        self,
        initial_prob: float = 0.1,
        max_prob: float = 0.8,
        initial_intensity: float = 0.1,
        max_intensity: float = 1.0,
        warmup_epochs: int = 5,
        total_epochs: int = 20,
        schedule_type: str = "cosine",  # "linear", "cosine", "exponential"
    ):
        self.initial_prob = initial_prob
        self.max_prob = max_prob
        self.initial_intensity = initial_intensity
        self.max_intensity = max_intensity
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0
    
    def get_probability(self, epoch: Optional[int] = None) -> float:
        """获取当前轮次的数据增强概率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增加
            progress = self.current_epoch / self.warmup_epochs
            prob = self.initial_prob + (self.max_prob - self.initial_prob) * progress
        else:
            # 主训练阶段
            remaining_epochs = self.total_epochs - self.warmup_epochs
            current_progress = (self.current_epoch - self.warmup_epochs) / remaining_epochs
            current_progress = min(1.0, current_progress)
            
            if self.schedule_type == "linear":
                prob = self.initial_prob + (self.max_prob - self.initial_prob) * current_progress
            elif self.schedule_type == "cosine":
                prob = self.initial_prob + (self.max_prob - self.initial_prob) * \
                       (1 - math.cos(math.pi * current_progress)) / 2
            elif self.schedule_type == "exponential":
                prob = self.initial_prob * (self.max_prob / self.initial_prob) ** current_progress
            else:
                prob = self.max_prob
        
        return max(self.initial_prob, min(self.max_prob, prob))
    
    def get_intensity(self, epoch: Optional[int] = None) -> float:
        """获取当前轮次的增强强度"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            intensity = self.initial_intensity + (self.max_intensity - self.initial_intensity) * progress
        else:
            remaining_epochs = self.total_epochs - self.warmup_epochs
            current_progress = (self.current_epoch - self.warmup_epochs) / remaining_epochs
            current_progress = min(1.0, current_progress)
            
            if self.schedule_type == "linear":
                intensity = self.initial_intensity + (self.max_intensity - self.initial_intensity) * current_progress
            elif self.schedule_type == "cosine":
                intensity = self.initial_intensity + (self.max_intensity - self.initial_intensity) * \
                           (1 - math.cos(math.pi * current_progress)) / 2
            elif self.schedule_type == "exponential":
                intensity = self.initial_intensity * (self.max_intensity / self.initial_intensity) ** current_progress
            else:
                intensity = self.max_intensity
        
        return max(self.initial_intensity, min(self.max_intensity, intensity))
    
    def reset(self):
        """重置调度器"""
        self.current_epoch = 0


class VideoNoiseAugmentation:
    """
    视频噪声增强类
    
    支持多种帧采样策略和噪声增强方法。
    """
    
    def __init__(
        self,
        target_num_frames: int = 8,
        frame_sampling_strategy: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM,
        # 空间噪声参数
        enable_spatial_noise: bool = True,
        gaussian_blur_prob: float = 0.3,
        occlusion_prob: float = 0.2,
        color_jitter_prob: float = 0.3,
        # 时序噪声参数
        enable_temporal_noise: bool = True,
        frame_repetition_prob: float = 0.2,
        frame_dropping_prob: float = 0.2,
        order_shuffle_prob: float = 0.3,
        # 语义噪声参数
        enable_semantic_noise: bool = True,
        irrelevant_frame_prob: float = 0.1,
        partial_occlusion_prob: float = 0.15,
        # 其他参数
        augmentation_prob_scheduler: Optional[AugmentationProbabilityScheduler] = None,
        device: Optional[torch.device] = None,
    ):
        self.target_num_frames = target_num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.enable_spatial_noise = enable_spatial_noise
        self.enable_temporal_noise = enable_temporal_noise
        self.enable_semantic_noise = enable_semantic_noise
        
        # 空间噪声概率
        self.gaussian_blur_prob = gaussian_blur_prob
        self.occlusion_prob = occlusion_prob
        self.color_jitter_prob = color_jitter_prob
        
        # 时序噪声概率
        self.frame_repetition_prob = frame_repetition_prob
        self.frame_dropping_prob = frame_dropping_prob
        self.order_shuffle_prob = order_shuffle_prob
        
        # 语义噪声概率
        self.irrelevant_frame_prob = irrelevant_frame_prob
        self.partial_occlusion_prob = partial_occlusion_prob
        
        # 概率调度器
        self.augmentation_prob_scheduler = augmentation_prob_scheduler
        
        # 设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化albumentations变换（如果可用）
        if ALBUMENTATIONS_AVAILABLE:
            self.albumentations_transform = A.Compose([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ])
        else:
            self.albumentations_transform = None
    
    def sample_frames(
        self,
        video: torch.Tensor,
        strategy: Optional[FrameSamplingStrategy] = None,
    ) -> torch.Tensor:
        """
        帧采样
        
        Args:
            video: 输入视频，形状为 (T, C, H, W)
            strategy: 采样策略，如果为None则使用self.frame_sampling_strategy
            
        Returns:
            采样后的视频帧，形状为 (target_num_frames, C, H, W)
        """
        if strategy is None:
            strategy = self.frame_sampling_strategy
        
        T, C, H, W = video.shape
        
        if T <= self.target_num_frames:
            # 如果帧数不足，直接返回或重复最后一帧
            if T < self.target_num_frames:
                padding = self.target_num_frames - T
                last_frame = video[-1:].repeat(padding, 1, 1, 1)
                video = torch.cat([video, last_frame], dim=0)
            return video
        
        if strategy == FrameSamplingStrategy.UNIFORM:
            # 均匀采样
            indices = torch.linspace(0, T - 1, self.target_num_frames).long()
            sampled_frames = video[indices]
        
        elif strategy == FrameSamplingStrategy.KEYFRAME:
            # 关键帧采样：使用帧间差异检测关键帧
            # 简化版本：选择变化较大的帧
            frame_diffs = []
            for i in range(1, T):
                diff = torch.mean((video[i] - video[i-1]) ** 2)
                frame_diffs.append(diff.item())
            
            # 选择差异最大的帧作为关键帧
            frame_diffs = torch.tensor(frame_diffs)
            _, top_indices = torch.topk(frame_diffs, min(self.target_num_frames - 1, len(frame_diffs)))
            top_indices = top_indices.sort()[0] + 1  # +1因为diff是从第二帧开始的
            indices = torch.cat([torch.tensor([0]), top_indices])
            if len(indices) < self.target_num_frames:
                # 补充均匀采样
                remaining = self.target_num_frames - len(indices)
                uniform_indices = torch.linspace(0, T - 1, remaining + 2)[1:-1].long()
                indices = torch.cat([indices, uniform_indices])
                indices = torch.unique(indices.sort()[0])[:self.target_num_frames]
            sampled_frames = video[indices]
        
        elif strategy == FrameSamplingStrategy.RANDOM:
            # 随机采样
            indices = torch.randperm(T)[:self.target_num_frames].sort()[0]
            sampled_frames = video[indices]
        
        else:
            # 默认均匀采样
            indices = torch.linspace(0, T - 1, self.target_num_frames).long()
            sampled_frames = video[indices]
        
        return sampled_frames
    
    def apply_spatial_noise(
        self,
        frames: torch.Tensor,
        intensity: float = 1.0,
    ) -> torch.Tensor:
        """
        应用空间噪声增强
        
        Args:
            frames: 输入帧，形状为 (T, C, H, W)
            intensity: 增强强度，范围[0, 1]
            
        Returns:
            增强后的帧，形状为 (T, C, H, W)
        """
        if not self.enable_spatial_noise:
            return frames
        
        T, C, H, W = frames.shape
        augmented_frames = frames.clone()
        
        # 获取当前增强概率
        prob = intensity
        if self.augmentation_prob_scheduler is not None:
            prob = self.augmentation_prob_scheduler.get_probability()
        
        for t in range(T):
            frame = frames[t]
            
            # 高斯模糊
            if random.random() < self.gaussian_blur_prob * prob:
                kernel_size = int(3 + intensity * 4)  # 3-7
                if kernel_size % 2 == 0:
                    kernel_size += 1
                sigma = 0.5 + intensity * 1.5
                frame = TF.gaussian_blur(frame, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            
            # 遮挡（随机矩形遮挡）
            if random.random() < self.occlusion_prob * prob:
                num_occlusions = random.randint(1, 3)
                for _ in range(num_occlusions):
                    occl_h = random.randint(int(H * 0.1), int(H * 0.3))
                    occl_w = random.randint(int(W * 0.1), int(W * 0.3))
                    occl_y = random.randint(0, H - occl_h)
                    occl_x = random.randint(0, W - occl_w)
                    # 使用随机颜色或黑色遮挡
                    if random.random() < 0.5:
                        frame[:, occl_y:occl_y+occl_h, occl_x:occl_x+occl_w] = 0.0
                    else:
                        frame[:, occl_y:occl_y+occl_h, occl_x:occl_x+occl_w] = torch.rand(C, 1, 1)
            
            # 色彩抖动
            if random.random() < self.color_jitter_prob * prob:
                brightness = 0.1 + intensity * 0.2
                contrast = 0.1 + intensity * 0.2
                saturation = 0.1 + intensity * 0.2
                hue = 0.05 + intensity * 0.05
                
                frame = TF.adjust_brightness(frame, 1.0 + random.uniform(-brightness, brightness))
                frame = TF.adjust_contrast(frame, 1.0 + random.uniform(-contrast, contrast))
                frame = TF.adjust_saturation(frame, 1.0 + random.uniform(-saturation, saturation))
                frame = TF.adjust_hue(frame, random.uniform(-hue, hue))
            
            augmented_frames[t] = frame
        
        return augmented_frames
    
    def apply_temporal_noise(
        self,
        frames: torch.Tensor,
        intensity: float = 1.0,
    ) -> torch.Tensor:
        """
        应用时序噪声增强
        
        Args:
            frames: 输入帧，形状为 (T, C, H, W)
            intensity: 增强强度，范围[0, 1]
            
        Returns:
            增强后的帧，形状为 (T, C, H, W)
        """
        if not self.enable_temporal_noise:
            return frames
        
        T, C, H, W = frames.shape
        augmented_frames = frames.clone()
        
        # 获取当前增强概率
        prob = intensity
        if self.augmentation_prob_scheduler is not None:
            prob = self.augmentation_prob_scheduler.get_probability()
        
        # 帧重复
        if random.random() < self.frame_repetition_prob * prob:
            num_repeats = random.randint(1, min(3, T // 2))
            repeat_indices = random.sample(range(T), num_repeats)
            for idx in repeat_indices:
                if idx < T - 1:
                    augmented_frames[idx + 1] = augmented_frames[idx]
        
        # 帧丢弃
        if random.random() < self.frame_dropping_prob * prob:
            num_drops = random.randint(1, min(2, T // 3))
            drop_indices = random.sample(range(T), num_drops)
            # 用相邻帧填充
            for idx in sorted(drop_indices, reverse=True):
                if idx < T - 1:
                    augmented_frames[idx] = augmented_frames[idx + 1]
                elif idx > 0:
                    augmented_frames[idx] = augmented_frames[idx - 1]
        
        # 顺序错乱
        if random.random() < self.order_shuffle_prob * prob:
            # 部分打乱顺序
            shuffle_length = max(2, int(T * (0.3 + intensity * 0.4)))
            start_idx = random.randint(0, max(0, T - shuffle_length))
            end_idx = start_idx + shuffle_length
            segment = augmented_frames[start_idx:end_idx]
            shuffled_indices = torch.randperm(len(segment))
            augmented_frames[start_idx:end_idx] = segment[shuffled_indices]
        
        return augmented_frames
    
    def apply_semantic_noise(
        self,
        frames: torch.Tensor,
        intensity: float = 1.0,
    ) -> torch.Tensor:
        """
        应用语义噪声增强
        
        Args:
            frames: 输入帧，形状为 (T, C, H, W)
            intensity: 增强强度，范围[0, 1]
            
        Returns:
            增强后的帧，形状为 (T, C, H, W)
        """
        if not self.enable_semantic_noise:
            return frames
        
        T, C, H, W = frames.shape
        augmented_frames = frames.clone()
        
        # 获取当前增强概率
        prob = intensity
        if self.augmentation_prob_scheduler is not None:
            prob = self.augmentation_prob_scheduler.get_probability()
        
        # 添加无关帧（使用噪声或模糊帧）
        if random.random() < self.irrelevant_frame_prob * prob:
            num_irrelevant = random.randint(1, min(2, T // 4))
            irrelevant_indices = random.sample(range(T), num_irrelevant)
            for idx in irrelevant_indices:
                # 生成噪声帧
                noise_frame = torch.randn_like(frames[idx]) * 0.5 + 0.5
                noise_frame = torch.clamp(noise_frame, 0.0, 1.0)
                augmented_frames[idx] = noise_frame
        
        # 部分遮挡关键物体（中心区域遮挡）
        if random.random() < self.partial_occlusion_prob * prob:
            num_occlusions = random.randint(1, min(2, T // 3))
            occlusion_indices = random.sample(range(T), num_occlusions)
            for idx in occlusion_indices:
                # 遮挡中心区域
                center_y, center_x = H // 2, W // 2
                occl_h = int(H * (0.2 + intensity * 0.2))
                occl_w = int(W * (0.2 + intensity * 0.2))
                y1 = max(0, center_y - occl_h // 2)
                y2 = min(H, center_y + occl_h // 2)
                x1 = max(0, center_x - occl_w // 2)
                x2 = min(W, center_x + occl_w // 2)
                # 使用模糊或噪声遮挡
                if random.random() < 0.5:
                    augmented_frames[idx, :, y1:y2, x1:x2] = 0.0
                else:
                    augmented_frames[idx, :, y1:y2, x1:x2] = torch.rand(C, y2-y1, x2-x1)
        
        return augmented_frames
    
    def create_batch(
        self,
        video: torch.Tensor,
        intensity: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        创建批次数据，包含干净、乱序、含噪三个版本
        
        Args:
            video: 输入视频，形状为 (T, C, H, W)
            intensity: 增强强度，如果为None则从调度器获取
            
        Returns:
            包含以下键的字典：
                - "clean": 干净版本
                - "shuffled": 乱序版本（仅时序打乱）
                - "noisy": 含噪版本（包含所有噪声）
        """
        # 帧采样
        sampled_frames = self.sample_frames(video)
        T, C, H, W = sampled_frames.shape
        
        # 获取增强强度
        if intensity is None:
            if self.augmentation_prob_scheduler is not None:
                intensity = self.augmentation_prob_scheduler.get_intensity()
            else:
                intensity = 1.0
        
        # 1. 干净版本（仅采样，不增强）
        clean_frames = sampled_frames.clone()
        
        # 2. 乱序版本（仅时序打乱）
        shuffled_indices = torch.randperm(T)
        shuffled_frames = sampled_frames[shuffled_indices]
        
        # 3. 含噪版本（应用所有噪声）
        noisy_frames = sampled_frames.clone()
        
        # 应用空间噪声
        if self.enable_spatial_noise:
            noisy_frames = self.apply_spatial_noise(noisy_frames, intensity)
        
        # 应用时序噪声
        if self.enable_temporal_noise:
            noisy_frames = self.apply_temporal_noise(noisy_frames, intensity)
        
        # 应用语义噪声
        if self.enable_semantic_noise:
            noisy_frames = self.apply_semantic_noise(noisy_frames, intensity)
        
        return {
            "clean": clean_frames,
            "shuffled": shuffled_frames,
            "noisy": noisy_frames,
        }
    
    def __call__(
        self,
        video: torch.Tensor,
        intensity: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """调用接口"""
        return self.create_batch(video, intensity)


def create_video_augmentation(
    target_num_frames: int = 8,
    frame_sampling_strategy: str = "uniform",
    enable_spatial_noise: bool = True,
    enable_temporal_noise: bool = True,
    enable_semantic_noise: bool = True,
    augmentation_prob_scheduler: Optional[AugmentationProbabilityScheduler] = None,
    device: Optional[torch.device] = None,
) -> VideoNoiseAugmentation:
    """
    创建视频噪声增强器的便捷函数
    
    Args:
        target_num_frames: 目标帧数
        frame_sampling_strategy: 帧采样策略，"uniform", "keyframe", "random"
        enable_spatial_noise: 是否启用空间噪声
        enable_temporal_noise: 是否启用时序噪声
        enable_semantic_noise: 是否启用语义噪声
        augmentation_prob_scheduler: 增强概率调度器
        device: 计算设备
        
    Returns:
        VideoNoiseAugmentation实例
    """
    strategy_map = {
        "uniform": FrameSamplingStrategy.UNIFORM,
        "keyframe": FrameSamplingStrategy.KEYFRAME,
        "random": FrameSamplingStrategy.RANDOM,
    }
    strategy = strategy_map.get(frame_sampling_strategy.lower(), FrameSamplingStrategy.UNIFORM)
    
    return VideoNoiseAugmentation(
        target_num_frames=target_num_frames,
        frame_sampling_strategy=strategy,
        enable_spatial_noise=enable_spatial_noise,
        enable_temporal_noise=enable_temporal_noise,
        enable_semantic_noise=enable_semantic_noise,
        augmentation_prob_scheduler=augmentation_prob_scheduler,
        device=device,
    )

