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
Selection Augmented Policy Implementation

This module implements a policy network that selects between clean and noisy frames
based on learned selection probabilities. It uses a pre-trained vision encoder (CLIP-ViT)
and a two-layer MLP selection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List
import warnings

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn(
        "CLIP is not available. Please install it using: pip install git+https://github.com/openai/CLIP.git"
    )


class SelectionAugmentedPolicy(nn.Module):
    """
    选择增强策略网络
    
    该网络学习在干净帧和含噪帧之间进行选择，以优化推理性能。
    使用预训练的CLIP视觉编码器提取特征，然后通过两层MLP输出选择概率。
    
    Args:
        vision_encoder_name (str): 视觉编码器名称，默认 "ViT-B/32"
        hidden_dim (int): MLP隐藏层维度，默认512
        dropout (float): Dropout概率，默认0.1
        freeze_vision_encoder (bool): 是否冻结视觉编码器，默认False
        selection_temperature (float): 选择概率的温度参数，默认1.0
        use_gumbel_softmax (bool): 是否使用Gumbel-Softmax进行可微采样，默认False
        detach_vision_features (bool): 是否断开视觉特征的梯度，默认False（用于节省显存）
    """
    
    def __init__(
        self,
        vision_encoder_name: str = "ViT-B/32",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_vision_encoder: bool = False,
        selection_temperature: float = 1.0,
        use_gumbel_softmax: bool = False,
        detach_vision_features: bool = False,
    ):
        super().__init__()
        
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP is required but not installed. "
                "Please install it using: pip install git+https://github.com/openai/CLIP.git"
            )
        
        # 加载预训练的CLIP视觉编码器
        self.vision_encoder, self.vision_preprocess = clip.load(vision_encoder_name, device="cpu")
        self.vision_encoder_name = vision_encoder_name
        
        # 获取视觉编码器的输出维度
        # CLIP的视觉编码器输出维度可以通过visual.proj或直接测试获取
        try:
            # 尝试从配置中获取
            if hasattr(self.vision_encoder.visual, 'output_dim'):
                vision_dim = self.vision_encoder.visual.output_dim
            elif hasattr(self.vision_encoder.visual, 'proj'):
                # 如果proj存在，输出维度是proj的输出维度
                vision_dim = self.vision_encoder.visual.proj.out_features
            else:
                # 默认值或通过测试获取
                test_input = torch.zeros(1, 3, 224, 224)
                with torch.no_grad():
                    test_output = self.vision_encoder.encode_image(test_input)
                vision_dim = test_output.shape[-1]
        except Exception:
            # 如果以上方法都失败，使用默认值
            # ViT-B/32的输出维度是512，ViT-L/14是768
            if "ViT-B" in vision_encoder_name or "vit-base" in vision_encoder_name.lower():
                vision_dim = 512
            elif "ViT-L" in vision_encoder_name or "vit-large" in vision_encoder_name.lower():
                vision_dim = 768
            else:
                vision_dim = 512  # 默认值
        
        # 冻结视觉编码器（如果需要）
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # 两层MLP选择头
        # 输入：clean特征 + noisy特征 + 特征差异
        # 输出：选择概率（2维：clean概率，noisy概率）
        input_dim = vision_dim * 3  # clean特征 + noisy特征 + 特征差异
        
        self.selection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # 输出2维：clean概率，noisy概率
        )
        
        # 配置参数
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.freeze_vision_encoder = freeze_vision_encoder
        self.selection_temperature = selection_temperature
        self.use_gumbel_softmax = use_gumbel_softmax
        self.detach_vision_features = detach_vision_features
        
        # 初始化选择头的权重
        self._init_selection_head()
    
    def _init_selection_head(self):
        """初始化选择头的权重"""
        for module in self.selection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def encode_frames(
        self,
        frames: torch.Tensor,
        detach: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        使用视觉编码器编码帧
        
        Args:
            frames (torch.Tensor): 输入帧，形状为 (B, T, C, H, W) 或 (B, C, H, W)
            detach (Optional[bool]): 是否断开梯度，如果为None则使用self.detach_vision_features
            
        Returns:
            torch.Tensor: 编码后的特征，形状为 (B, T, D) 或 (B, D)
        """
        if detach is None:
            detach = self.detach_vision_features
        
        # 处理输入维度
        original_shape = frames.shape
        if frames.dim() == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            B, T, C, H, W = frames.shape
            frames = frames.view(B * T, C, H, W)
            reshape_output = True
        elif frames.dim() == 4:
            # (B, C, H, W)
            B = frames.shape[0]
            T = 1
            reshape_output = False
        else:
            raise ValueError(f"Expected 4D or 5D input, got {frames.dim()}D")
        
        # 如果需要在视觉编码器前断开梯度
        if detach:
            frames = frames.detach()
        
        # 使用CLIP视觉编码器提取特征
        # CLIP的encode_image期望输入为 (B, C, H, W)，且值在[0, 1]范围内
        # 如果输入不在[0, 1]范围内，需要归一化
        if frames.max() > 1.0:
            # 假设输入是[0, 255]范围，归一化到[0, 1]
            frames = frames / 255.0
        elif frames.min() < 0.0:
            # 如果输入是[-1, 1]范围，归一化到[0, 1]
            frames = (frames + 1.0) / 2.0
        
        with torch.set_grad_enabled(not detach):
            features = self.vision_encoder.encode_image(frames)
        
        # 如果需要在视觉编码器后断开梯度
        if detach:
            features = features.detach()
        
        # 重塑输出
        if reshape_output:
            # (B*T, D) -> (B, T, D)
            features = features.view(B, T, -1)
        
        return features
    
    def compute_selection_probabilities(
        self,
        clean_features: torch.Tensor,
        noisy_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算选择概率
        
        Args:
            clean_features (torch.Tensor): 干净帧的特征，形状为 (B, T, D) 或 (B, D)
            noisy_features (torch.Tensor): 含噪帧的特征，形状为 (B, T, D) 或 (B, D)
            
        Returns:
            torch.Tensor: 选择概率，形状为 (B, T, 2) 或 (B, 2)，最后一维为 [clean_prob, noisy_prob]
        """
        # 计算特征差异
        feature_diff = clean_features - noisy_features
        
        # 展平特征（如果是时序数据）
        if clean_features.dim() == 3:
            # (B, T, D) -> (B*T, D)
            B, T, D = clean_features.shape
            clean_features_flat = clean_features.view(B * T, D)
            noisy_features_flat = noisy_features.view(B * T, D)
            feature_diff_flat = feature_diff.view(B * T, D)
            reshape_output = True
        else:
            # (B, D)
            clean_features_flat = clean_features
            noisy_features_flat = noisy_features
            feature_diff_flat = feature_diff
            reshape_output = False
        
        # 拼接特征
        combined_features = torch.cat([
            clean_features_flat,
            noisy_features_flat,
            feature_diff_flat,
        ], dim=-1)
        
        # 通过选择头
        logits = self.selection_head(combined_features)
        
        # 应用温度缩放
        if self.selection_temperature != 1.0:
            logits = logits / self.selection_temperature
        
        # 计算概率
        if self.use_gumbel_softmax and self.training:
            # 使用Gumbel-Softmax进行可微采样
            probs = F.gumbel_softmax(logits, tau=self.selection_temperature, hard=False, dim=-1)
        else:
            # 使用标准softmax
            probs = F.softmax(logits, dim=-1)
        
        # 重塑输出
        if reshape_output:
            # (B*T, 2) -> (B, T, 2)
            probs = probs.view(B, T, 2)
        
        return probs
    
    def select_features(
        self,
        clean_features: torch.Tensor,
        noisy_features: torch.Tensor,
        selection_probs: torch.Tensor,
        use_hard_selection: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据选择概率选择特征
        
        Args:
            clean_features (torch.Tensor): 干净帧的特征
            noisy_features (torch.Tensor): 含噪帧的特征
            selection_probs (torch.Tensor): 选择概率，形状为 (B, T, 2) 或 (B, 2)
            use_hard_selection (bool): 是否使用硬选择（argmax），默认False（使用软选择）
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 选择的特征
                - 选择类型（0表示clean，1表示noisy）
        """
        if use_hard_selection:
            # 硬选择：使用argmax
            selection_indices = selection_probs.argmax(dim=-1)  # (B, T) 或 (B,)
            selection_types = selection_indices
        else:
            # 软选择：使用加权平均
            # selection_probs: (B, T, 2) 或 (B, 2)
            # 最后一维：[clean_prob, noisy_prob]
            if selection_probs.dim() == 3:
                # (B, T, 2)
                clean_weight = selection_probs[..., 0:1]  # (B, T, 1)
                noisy_weight = selection_probs[..., 1:2]  # (B, T, 1)
                selection_types = selection_probs.argmax(dim=-1)  # (B, T)
            else:
                # (B, 2)
                clean_weight = selection_probs[:, 0:1]  # (B, 1)
                noisy_weight = selection_probs[:, 1:2]  # (B, 1)
                selection_types = selection_probs.argmax(dim=-1)  # (B,)
            
            # 加权组合特征
            selected_features = clean_weight * clean_features + noisy_weight * noisy_features
        
        if use_hard_selection:
            # 硬选择：根据索引选择
            if clean_features.dim() == 3:
                # (B, T, D)
                B, T, D = clean_features.shape
                batch_indices = torch.arange(B, device=clean_features.device).unsqueeze(1).expand(-1, T)
                time_indices = torch.arange(T, device=clean_features.device).unsqueeze(0).expand(B, -1)
                selected_features = torch.where(
                    selection_indices.unsqueeze(-1) == 0,
                    clean_features,
                    noisy_features,
                )
            else:
                # (B, D)
                selected_features = torch.where(
                    selection_indices.unsqueeze(-1) == 0,
                    clean_features,
                    noisy_features,
                )
        
        return selected_features, selection_types
    
    def get_selection_confidence(
        self,
        selection_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算选择置信度
        
        置信度定义为选择概率的最大值与最小值的差异，范围在[0, 1]。
        值越大表示选择越确定。
        
        Args:
            selection_probs (torch.Tensor): 选择概率，形状为 (B, T, 2) 或 (B, 2)
            
        Returns:
            torch.Tensor: 选择置信度，形状为 (B, T) 或 (B)
        """
        # 计算最大概率和最小概率的差异
        max_probs = selection_probs.max(dim=-1)[0]
        min_probs = selection_probs.min(dim=-1)[0]
        confidence = max_probs - min_probs
        
        return confidence
    
    def forward(
        self,
        clean_frames: torch.Tensor,
        noisy_frames: torch.Tensor,
        question: Optional[Union[str, List[str]]] = None,
        detach_vision: Optional[bool] = None,
        use_hard_selection: bool = False,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        实现条件特征选择逻辑：根据学习到的选择概率在干净帧和含噪帧之间进行选择。
        
        Args:
            clean_frames (torch.Tensor): 干净帧，形状为 (B, T, C, H, W) 或 (B, C, H, W)
            noisy_frames (torch.Tensor): 含噪帧，形状为 (B, T, C, H, W) 或 (B, C, H, W)
            question (Optional[Union[str, List[str]]]): 问题文本（当前未使用，保留用于未来扩展）
            detach_vision (Optional[bool]): 是否断开视觉特征的梯度，如果为None则使用self.detach_vision_features
            use_hard_selection (bool): 是否使用硬选择，默认False
            return_intermediates (bool): 是否返回中间结果，默认False
            
        Returns:
            Dict[str, torch.Tensor]: 包含以下键的字典
                - "selected_features": 选择的特征
                - "selection_probs": 选择概率，形状为 (B, T, 2) 或 (B, 2)
                - "selection_types": 选择类型（0=clean, 1=noisy），形状为 (B, T) 或 (B)
                - "selection_confidence": 选择置信度，形状为 (B, T) 或 (B)
                - "clean_features": 干净帧特征（如果return_intermediates=True）
                - "noisy_features": 含噪帧特征（如果return_intermediates=True）
        """
        # 编码干净帧和含噪帧
        clean_features = self.encode_frames(clean_frames, detach=detach_vision)
        noisy_features = self.encode_frames(noisy_frames, detach=detach_vision)
        
        # 计算选择概率
        selection_probs = self.compute_selection_probabilities(clean_features, noisy_features)
        
        # 根据选择概率选择特征
        selected_features, selection_types = self.select_features(
            clean_features,
            noisy_features,
            selection_probs,
            use_hard_selection=use_hard_selection,
        )
        
        # 计算选择置信度
        selection_confidence = self.get_selection_confidence(selection_probs)
        
        # 构建输出字典
        output = {
            "selected_features": selected_features,
            "selection_probs": selection_probs,
            "selection_types": selection_types,
            "selection_confidence": selection_confidence,
        }
        
        # 如果需要返回中间结果
        if return_intermediates:
            output["clean_features"] = clean_features
            output["noisy_features"] = noisy_features
        
        return output
    
    def to_device(self, device: Union[str, torch.device]):
        """
        将模型移动到指定设备
        
        Args:
            device (Union[str, torch.device]): 目标设备
        """
        self.to(device)
        # CLIP模型也需要移动到设备
        if hasattr(self.vision_encoder, 'to'):
            self.vision_encoder = self.vision_encoder.to(device)
    
    def train(self, mode: bool = True):
        """
        设置训练/评估模式
        
        Args:
            mode (bool): 如果True，设置为训练模式；如果False，设置为评估模式
        """
        super().train(mode)
        # 如果视觉编码器被冻结，确保它处于评估模式
        if self.freeze_vision_encoder:
            self.vision_encoder.eval()
        return self


def create_selection_policy(
    vision_encoder_name: str = "ViT-B/32",
    hidden_dim: int = 512,
    dropout: float = 0.1,
    freeze_vision_encoder: bool = False,
    selection_temperature: float = 1.0,
    use_gumbel_softmax: bool = False,
    detach_vision_features: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> SelectionAugmentedPolicy:
    """
    创建选择增强策略网络的便捷函数
    
    Args:
        vision_encoder_name (str): 视觉编码器名称
        hidden_dim (int): MLP隐藏层维度
        dropout (float): Dropout概率
        freeze_vision_encoder (bool): 是否冻结视觉编码器
        selection_temperature (float): 选择概率的温度参数
        use_gumbel_softmax (bool): 是否使用Gumbel-Softmax
        detach_vision_features (bool): 是否断开视觉特征的梯度
        device (Optional[Union[str, torch.device]]): 目标设备
        
    Returns:
        SelectionAugmentedPolicy: 初始化好的策略网络
    """
    policy = SelectionAugmentedPolicy(
        vision_encoder_name=vision_encoder_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_vision_encoder=freeze_vision_encoder,
        selection_temperature=selection_temperature,
        use_gumbel_softmax=use_gumbel_softmax,
        detach_vision_features=detach_vision_features,
    )
    
    if device is not None:
        policy.to_device(device)
    
    return policy

