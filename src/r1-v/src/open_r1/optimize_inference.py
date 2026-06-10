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
Inference Optimization Script

Implements various inference optimizations:
1. Model quantization (INT8/FP16)
2. Frame caching mechanism to avoid repeated encoding
3. Dynamic batching
4. Selection head early-exit (skip full inference for high confidence)
5. Distributed inference support
6. Benchmarking inference speed and memory usage
"""

import os
import time
import hashlib
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from queue import Queue

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, some features may be limited")

try:
    import torch.quantization
    import torch.nn.quantized
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: Quantization not fully available")

from trainer import SelectionAugmentedPolicy


@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str
    quantization: Optional[str] = None  # "int8", "fp16", "bf16", None
    use_cache: bool = True
    cache_size: int = 1000
    dynamic_batching: bool = True
    max_batch_size: int = 8
    early_exit_threshold: float = 0.95
    use_early_exit: bool = True
    distributed: bool = False
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    use_vllm: bool = True
    max_model_len: int = 81920
    gpu_memory_utilization: float = 0.8


class FrameCache:
    """
    帧缓存机制
    
    避免重复编码相同的视频帧。
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def _get_frame_hash(self, frame_path: str, frame_idx: int = None) -> str:
        """生成帧的唯一标识符"""
        if frame_idx is not None:
            key = f"{frame_path}:{frame_idx}"
        else:
            key = frame_path
        
        # 使用文件路径和修改时间生成哈希
        try:
            stat = os.stat(frame_path)
            mtime = stat.st_mtime
            return hashlib.md5(f"{key}:{mtime}".encode()).hexdigest()
        except:
            return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, frame_path: str, frame_idx: int = None) -> Optional[torch.Tensor]:
        """获取缓存的帧特征"""
        with self.lock:
            key = self._get_frame_hash(frame_path, frame_idx)
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key].clone()
            return None
    
    def put(self, frame_path: str, features: torch.Tensor, frame_idx: int = None):
        """存储帧特征到缓存"""
        with self.lock:
            key = self._get_frame_hash(frame_path, frame_idx)
            
            # 如果缓存已满，删除最久未使用的
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = features.detach().cpu()
            self.access_times[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1),
            }


class DynamicBatcher:
    """
    动态批处理器
    
    根据输入大小动态调整批次大小。
    """
    
    def __init__(self, max_batch_size: int = 8, max_tokens: int = 8192):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.queue: Queue = Queue()
    
    def add_request(self, request: Dict[str, Any]):
        """添加请求到队列"""
        self.queue.put(request)
    
    def get_batch(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """
        获取一个批次
        
        Args:
            timeout: 等待超时时间（秒）
        
        Returns:
            批次请求列表
        """
        batch = []
        total_tokens = 0
        
        # 获取第一个请求
        try:
            first_request = self.queue.get(timeout=timeout)
            batch.append(first_request)
            total_tokens += first_request.get('num_tokens', 0)
        except:
            return batch
        
        # 尝试添加更多请求
        while len(batch) < self.max_batch_size:
            try:
                request = self.queue.get_nowait()
                request_tokens = request.get('num_tokens', 0)
                
                if total_tokens + request_tokens <= self.max_tokens:
                    batch.append(request)
                    total_tokens += request_tokens
                else:
                    # 放回队列
                    self.queue.put(request)
                    break
            except:
                break
        
        return batch


class OptimizedInferenceEngine:
    """
    优化的推理引擎
    
    集成所有优化技术。
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self._load_model()
        
        # 初始化缓存
        self.frame_cache = FrameCache(max_size=config.cache_size) if config.use_cache else None
        
        # 初始化批处理器
        self.batcher = DynamicBatcher(
            max_batch_size=config.max_batch_size,
        ) if config.dynamic_batching else None
        
        # 初始化选择策略（如果使用early-exit）
        self.selection_policy = None
        if config.use_early_exit:
            self.selection_policy = SelectionAugmentedPolicy(
                vision_encoder_name="ViT-B/32",
                freeze_vision_encoder=True,
                detach_vision_features=True,
            ).to(self.device).eval()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'early_exits': 0,
            'total_time': 0.0,
            'inference_time': 0.0,
            'cache_time': 0.0,
        }
    
    def _load_model(self):
        """加载并量化模型"""
        print(f"Loading model from {self.config.model_path}...")
        
        if self.config.use_vllm and VLLM_AVAILABLE:
            # 使用vLLM
            self.llm = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype=self.config.quantization or "auto",
                limit_mm_per_prompt={"video": 1, "image": 1},
            )
            self.processor = AutoProcessor.from_pretrained(self.config.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.tokenizer.padding_side = "left"
            self.processor.tokenizer = self.tokenizer
            self.use_vllm = True
        else:
            # 使用transformers
            model_class = Qwen2VLForConditionalGeneration
            try:
                self.model = model_class.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self._get_dtype(),
                    device_map="auto" if self.config.distributed else None,
                )
            except:
                # 尝试使用Qwen2_5_VL
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self._get_dtype(),
                    device_map="auto" if self.config.distributed else None,
                )
            
            self.model = self.model.to(self.device)
            
            # 量化
            if self.config.quantization == "int8" and QUANTIZATION_AVAILABLE:
                print("Applying INT8 quantization...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
            elif self.config.quantization == "fp16":
                self.model = self.model.half()
            elif self.config.quantization == "bf16":
                self.model = self.model.bfloat16()
            
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.use_vllm = False
        
        print("Model loaded successfully!")
    
    def _get_dtype(self) -> torch.dtype:
        """获取数据类型"""
        if self.config.quantization == "fp16":
            return torch.float16
        elif self.config.quantization == "bf16":
            return torch.bfloat16
        else:
            return torch.float32
    
    def _encode_frames_cached(
        self,
        video_path: str,
        frame_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        使用缓存编码帧
        
        Args:
            video_path: 视频路径
            frame_indices: 帧索引列表
        
        Returns:
            编码后的特征
        """
        if not self.frame_cache:
            # 如果没有缓存，直接编码
            return self._encode_frames(video_path, frame_indices)
        
        # 尝试从缓存获取
        if frame_indices:
            # 逐个帧检查缓存
            cached_features = []
            uncached_indices = []
            
            for idx in frame_indices:
                self._cache_requests += 1
                cached = self.frame_cache.get(video_path, idx)
                if cached is not None:
                    cached_features.append((idx, cached))
                    self.stats['cache_hits'] += 1
                    self._cache_hits += 1
                else:
                    uncached_indices.append(idx)
            
            # 编码未缓存的帧
            if uncached_indices:
                features = self._encode_frames(video_path, uncached_indices)
                # 存储到缓存
                for idx, feat in zip(uncached_indices, features):
                    self.frame_cache.put(video_path, feat, idx)
                # 合并特征
                all_features = sorted(cached_features + [(idx, feat) for idx, feat in zip(uncached_indices, features)])
                return torch.stack([feat for _, feat in all_features])
            else:
                # 所有帧都在缓存中
                all_features = sorted(cached_features)
                return torch.stack([feat for _, feat in all_features])
        else:
            # 整个视频
            self._cache_requests += 1
            cached = self.frame_cache.get(video_path)
            if cached is not None:
                self.stats['cache_hits'] += 1
                self._cache_hits += 1
                return cached
            
            features = self._encode_frames(video_path, frame_indices)
            self.frame_cache.put(video_path, features)
            return features
    
    def _encode_frames(
        self,
        video_path: str,
        frame_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        编码视频帧（需要实现具体逻辑）
        
        这里简化处理，实际应该加载视频并提取帧
        """
        # 这里应该实现实际的帧编码逻辑
        # 简化版本：返回随机特征
        num_frames = len(frame_indices) if frame_indices else 8
        return torch.randn(num_frames, 512).to(self.device)
    
    def _check_early_exit(
        self,
        clean_frames: torch.Tensor,
        noisy_frames: torch.Tensor,
        question: str,
    ) -> Tuple[bool, float, str]:
        """
        检查是否可以使用early-exit
        
        Returns:
            (should_exit, confidence, selection_type)
        """
        if not self.selection_policy:
            return False, 0.0, "none"
        
        with torch.no_grad():
            result = self.selection_policy(
                clean_frames=clean_frames,
                noisy_frames=noisy_frames,
                question=question,
            )
            
            selection_probs = result['selection_probs']
            confidence = torch.max(selection_probs).item()
            
            if confidence >= self.config.early_exit_threshold:
                selection_type = "clean" if torch.argmax(selection_probs).item() == 0 else "noisy"
                return True, confidence, selection_type
        
        return False, confidence, "none"
    
    def infer(
        self,
        video_path: str,
        question: str,
        clean_frames: Optional[torch.Tensor] = None,
        noisy_frames: Optional[torch.Tensor] = None,
        use_early_exit: bool = True,
    ) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            video_path: 视频路径
            question: 问题文本
            clean_frames: 干净帧（可选）
            noisy_frames: 含噪帧（可选）
            use_early_exit: 是否使用early-exit
        
        Returns:
            推理结果字典
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # Early-exit检查
        if use_early_exit and self.config.use_early_exit and clean_frames is not None and noisy_frames is not None:
            cache_start = time.time()
            should_exit, confidence, selection_type = self._check_early_exit(
                clean_frames, noisy_frames, question
            )
            self.stats['cache_time'] += time.time() - cache_start
            
            if should_exit:
                self.stats['early_exits'] += 1
                # 使用高置信度的选择，跳过完整推理
                selected_frames = clean_frames if selection_type == "clean" else noisy_frames
                # 这里可以返回一个简化的结果
                return {
                    'output': f"[Early-exit: {selection_type} with confidence {confidence:.3f}]",
                    'early_exit': True,
                    'confidence': confidence,
                    'selection_type': selection_type,
                    'inference_time': time.time() - start_time,
                }
        
        # 完整推理
        inference_start = time.time()
        
        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 200704,
                    "nframes": 32,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }]
        
        if self.use_vllm:
            # 使用vLLM推理
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            llm_inputs = [{
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs[0]},
                "mm_processor_kwargs": {key: val[0] for key, val in video_kwargs.items()},
            }]
            
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                max_tokens=1024,
            )
            
            outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
        else:
            # 使用transformers推理
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        inference_time = time.time() - inference_start
        self.stats['inference_time'] += inference_time
        self.stats['total_time'] += time.time() - start_time
        
        return {
            'output': output_text,
            'early_exit': False,
            'inference_time': inference_time,
            'total_time': time.time() - start_time,
        }
    
    def infer_batch(
        self,
        requests: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            requests: 请求列表，每个请求包含video_path和question
        
        Returns:
            结果列表
        """
        if self.use_vllm:
            # vLLM自动批处理
            messages_list = []
            for req in requests:
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": req['video_path'],
                            "max_pixels": 200704,
                            "nframes": 32,
                        },
                        {
                            "type": "text",
                            "text": req['question'],
                        },
                    ],
                }]
                messages_list.append(messages)
            
            prompts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_list
            ]
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages_list, return_video_kwargs=True)
            
            llm_inputs = []
            for idx, prompt in enumerate(prompts):
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"video": video_inputs[idx]},
                    "mm_processor_kwargs": {key: val[idx] for key, val in video_kwargs.items()},
                })
            
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                max_tokens=1024,
            )
            
            outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)
            results = [{'output': out.outputs[0].text} for out in outputs]
        else:
            # 手动批处理
            results = []
            for req in requests:
                result = self.infer(req['video_path'], req['question'])
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        if self.frame_cache:
            cache_stats = self.frame_cache.stats()
            # 更新缓存命中率
            if self._cache_requests > 0:
                cache_stats['hit_rate'] = self._cache_hits / self._cache_requests
            stats.update(cache_stats)
        
        if stats['total_requests'] > 0:
            stats['avg_inference_time'] = stats['inference_time'] / stats['total_requests']
            stats['avg_total_time'] = stats['total_time'] / stats['total_requests']
            if self._cache_requests > 0:
                stats['cache_hit_rate'] = self._cache_hits / self._cache_requests
            else:
                stats['cache_hit_rate'] = 0.0
            stats['early_exit_rate'] = stats['early_exits'] / stats['total_requests']
        
        return stats


def benchmark_inference(
    engine: OptimizedInferenceEngine,
    test_data: List[Dict[str, str]],
    num_runs: int = 5,
) -> Dict[str, Any]:
    """
    基准测试推理速度和内存使用
    
    Args:
        engine: 推理引擎
        test_data: 测试数据
        num_runs: 运行次数
    
    Returns:
        基准测试结果
    """
    print(f"Running benchmark with {len(test_data)} samples, {num_runs} runs each...")
    
    # 清空统计
    engine.stats = {
        'total_requests': 0,
        'cache_hits': 0,
        'early_exits': 0,
        'total_time': 0.0,
        'inference_time': 0.0,
        'cache_time': 0.0,
    }
    if engine.frame_cache:
        engine.frame_cache.clear()
    
    # 内存使用
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    
    # 预热
    print("Warming up...")
    for sample in test_data[:2]:
        engine.infer(sample['video_path'], sample['question'], use_early_exit=False)
    
    # 运行测试
    all_times = []
    for run in range(num_runs):
        run_times = []
        start_time = time.time()
        
        for sample in test_data:
            result = engine.infer(sample['video_path'], sample['question'])
            run_times.append(result.get('inference_time', 0))
        
        total_time = time.time() - start_time
        all_times.append(total_time)
        print(f"Run {run+1}/{num_runs}: {total_time:.2f}s ({total_time/len(test_data)*1000:.2f}ms per sample)")
    
    # 内存使用
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
    else:
        memory_used = 0
    
    # 统计信息
    stats = engine.get_stats()
    
    results = {
        'num_samples': len(test_data),
        'num_runs': num_runs,
        'total_time_mean': np.mean(all_times),
        'total_time_std': np.std(all_times),
        'time_per_sample_mean': np.mean(all_times) / len(test_data),
        'time_per_sample_std': np.std(all_times) / len(test_data),
        'throughput': len(test_data) / np.mean(all_times),
        'memory_used_mb': memory_used / 1024 / 1024,
        'stats': stats,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Optimized Inference Engine")
    parser.add_argument('--model-path', type=str, required=True, help='Model path')
    parser.add_argument('--quantization', type=str, choices=['int8', 'fp16', 'bf16', None], default=None)
    parser.add_argument('--use-cache', action='store_true', help='Use frame caching')
    parser.add_argument('--cache-size', type=int, default=1000, help='Cache size')
    parser.add_argument('--dynamic-batching', action='store_true', help='Use dynamic batching')
    parser.add_argument('--max-batch-size', type=int, default=8, help='Max batch size')
    parser.add_argument('--early-exit', action='store_true', help='Use early-exit')
    parser.add_argument('--early-exit-threshold', type=float, default=0.95, help='Early-exit confidence threshold')
    parser.add_argument('--distributed', action='store_true', help='Use distributed inference')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--use-vllm', action='store_true', help='Use vLLM')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--test-data', type=str, help='Path to test data JSON file')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # 创建配置
    config = InferenceConfig(
        model_path=args.model_path,
        quantization=args.quantization,
        use_cache=args.use_cache,
        cache_size=args.cache_size,
        dynamic_batching=args.dynamic_batching,
        max_batch_size=args.max_batch_size,
        use_early_exit=args.early_exit,
        early_exit_threshold=args.early_exit_threshold,
        distributed=args.distributed,
        tensor_parallel_size=args.tensor_parallel_size,
        use_vllm=args.use_vllm,
    )
    
    # 创建推理引擎
    engine = OptimizedInferenceEngine(config)
    
    if args.benchmark:
        # 加载测试数据
        if args.test_data:
            with open(args.test_data, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        else:
            # 使用示例数据
            test_data = [
                {'video_path': 'test_video.mp4', 'question': 'What is happening in this video?'},
            ] * 10
        
        # 运行基准测试
        results = benchmark_inference(engine, test_data, num_runs=args.num_runs)
        
        # 打印结果
        print("\n" + "="*80)
        print("Benchmark Results")
        print("="*80)
        print(f"Number of samples: {results['num_samples']}")
        print(f"Number of runs: {results['num_runs']}")
        print(f"Total time: {results['total_time_mean']:.2f} ± {results['total_time_std']:.2f} seconds")
        print(f"Time per sample: {results['time_per_sample_mean']*1000:.2f} ± {results['time_per_sample_std']*1000:.2f} ms")
        print(f"Throughput: {results['throughput']:.2f} samples/second")
        print(f"Memory used: {results['memory_used_mb']:.2f} MB")
        print(f"\nStatistics:")
        for key, value in results['stats'].items():
            print(f"  {key}: {value}")
        print("="*80)
        
        # 保存结果
        output_path = "benchmark_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    else:
        # 交互式推理
        print("Interactive inference mode. Enter 'quit' to exit.")
        while True:
            video_path = input("Video path: ")
            if video_path == 'quit':
                break
            question = input("Question: ")
            
            result = engine.infer(video_path, question)
            print(f"Output: {result['output']}")
            print(f"Inference time: {result.get('inference_time', 0):.3f}s")


if __name__ == "__main__":
    main()

