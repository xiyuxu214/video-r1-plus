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
Ablation Study Script

Automatically generates ablation experiment configurations, runs experiments,
and aggregates results into comparison tables.

Ablation studies:
1. Without selection mechanism (wo_selection)
2. Without robust reward (wo_robust_reward)
3. Without temporal comparison (wo_temporal)
4. Without image data (wo_image)
5. Different noise types (spatial_only, temporal_only, combined)
6. Different selection head architectures (MLP, Transformer, LSTM)
"""

import os
import json
import yaml
import subprocess
import argparse
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from trainer import ExperimentComparator, generate_comparison_table


class AblationStudyGenerator:
    """
    消融研究生成器
    
    自动生成消融实验配置并运行实验。
    """
    
    def __init__(
        self,
        base_config_path: str = "./config/experiment/base.yaml",
        output_dir: str = "./experiments/ablation",
        config_dir: str = "./config/experiment/ablation",
    ):
        self.base_config_path = base_config_path
        self.output_dir = output_dir
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载基础配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
    
    def generate_ablation_configs(self) -> Dict[str, str]:
        """
        生成所有消融实验配置
        
        Returns:
            字典，键为实验名称，值为配置文件路径
        """
        configs = {}
        
        # 1. 移除选择机制 (wo_selection)
        configs['wo_selection'] = self._create_config(
            name='wo_selection',
            description='Ablation: Without selection mechanism',
            overrides={
                'model': {'selection_policy': {'enabled': False}},
                'training': {'rewards': {'selection_weight': 0.0}},
            },
            tags=['ablation', 'wo-selection'],
        )
        
        # 2. 移除鲁棒奖励 (wo_robust_reward)
        configs['wo_robust_reward'] = self._create_config(
            name='wo_robust_reward',
            description='Ablation: Without robust reward',
            overrides={
                'training': {'rewards': {'robustness_penalty_weight': 0.0}},
            },
            tags=['ablation', 'wo-robust-reward'],
        )
        
        # 3. 移除时序对比 (wo_temporal)
        configs['wo_temporal'] = self._create_config(
            name='wo_temporal',
            description='Ablation: Without temporal comparison',
            overrides={
                'training': {
                    'temporal': {'enabled': False},
                    'rewards': {'temporal_weight': 0.0},
                },
            },
            tags=['ablation', 'wo-temporal'],
        )
        
        # 4. 移除图像数据 (wo_image)
        configs['wo_image'] = self._create_config(
            name='wo_image',
            description='Ablation: Without image data (video only)',
            overrides={
                'data': {'dataset': {'filter_image': True}},
            },
            tags=['ablation', 'wo-image', 'video-only'],
        )
        
        # 5. 不同噪声类型对比
        # 5.1 仅空间噪声
        configs['spatial_only'] = self._create_config(
            name='spatial_only',
            description='Ablation: Spatial noise only',
            overrides={
                'data': {
                    'video_augmentation': {
                        'spatial_noise': {'enabled': True},
                        'temporal_noise': {'enabled': False},
                        'semantic_noise': {'enabled': False},
                    }
                },
            },
            tags=['ablation', 'spatial-noise-only'],
        )
        
        # 5.2 仅时序噪声
        configs['temporal_only'] = self._create_config(
            name='temporal_only',
            description='Ablation: Temporal noise only',
            overrides={
                'data': {
                    'video_augmentation': {
                        'spatial_noise': {'enabled': False},
                        'temporal_noise': {'enabled': True},
                        'semantic_noise': {'enabled': False},
                    }
                },
            },
            tags=['ablation', 'temporal-noise-only'],
        )
        
        # 5.3 组合噪声（默认，但明确标记）
        configs['combined_noise'] = self._create_config(
            name='combined_noise',
            description='Ablation: Combined noise (spatial + temporal + semantic)',
            overrides={
                'data': {
                    'video_augmentation': {
                        'spatial_noise': {'enabled': True},
                        'temporal_noise': {'enabled': True},
                        'semantic_noise': {'enabled': True},
                    }
                },
            },
            tags=['ablation', 'combined-noise'],
        )
        
        # 6. 不同选择头架构对比
        # 6.1 MLP (默认)
        configs['selection_mlp'] = self._create_config(
            name='selection_mlp',
            description='Ablation: Selection head with MLP architecture',
            overrides={
                'model': {
                    'selection_policy': {
                        'selection_head': {
                            'architecture': 'mlp',
                            'hidden_dim': 512,
                        }
                    }
                },
            },
            tags=['ablation', 'selection-mlp'],
        )
        
        # 6.2 Transformer
        configs['selection_transformer'] = self._create_config(
            name='selection_transformer',
            description='Ablation: Selection head with Transformer architecture',
            overrides={
                'model': {
                    'selection_policy': {
                        'selection_head': {
                            'architecture': 'transformer',
                            'hidden_dim': 512,
                            'num_layers': 2,
                            'num_heads': 8,
                        }
                    }
                },
            },
            tags=['ablation', 'selection-transformer'],
        )
        
        # 6.3 LSTM
        configs['selection_lstm'] = self._create_config(
            name='selection_lstm',
            description='Ablation: Selection head with LSTM architecture',
            overrides={
                'model': {
                    'selection_policy': {
                        'selection_head': {
                            'architecture': 'lstm',
                            'hidden_dim': 512,
                            'num_layers': 2,
                        }
                    }
                },
            },
            tags=['ablation', 'selection-lstm'],
        )
        
        return configs
    
    def _create_config(
        self,
        name: str,
        description: str,
        overrides: Dict[str, Any],
        tags: List[str],
    ) -> str:
        """
        创建消融实验配置文件
        
        Args:
            name: 实验名称
            description: 实验描述
            overrides: 要覆盖的配置项
            tags: 标签列表
        
        Returns:
            配置文件路径
        """
        # 构建配置字典
        config = {
            'defaults': ['../base', '_self_'],
            'experiment': {
                'name': f'robust-grpo-{name}',
                'description': description,
                'tags': ['robust', 'grpo', 'video'] + tags,
                'output_dir': f'./experiments/ablation/{name}',
            },
        }
        
        # 合并覆盖配置
        self._merge_dict(config, overrides)
        
        # 保存配置文件
        config_path = os.path.join(self.config_dir, f'{name}.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            # 写入注释
            f.write(f"# Ablation Study: {description}\n")
            f.write("# This configuration is automatically generated\n\n")
            # 写入YAML内容
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"Generated ablation config: {config_path}")
        return config_path
    
    def _merge_dict(self, base: Dict, updates: Dict):
        """递归合并字典"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            else:
                base[key] = value
    
    def run_experiment(
        self,
        config_name: str,
        config_path: str,
        train_script: str = "train_robust_grpo_hydra.py",
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        运行单个消融实验
        
        Args:
            config_name: 实验名称
            config_path: 配置文件路径
            train_script: 训练脚本路径
            dry_run: 是否为试运行（不实际执行）
        
        Returns:
            实验结果字典，如果dry_run为True则返回None
        """
        print(f"\n{'='*80}")
        print(f"Running ablation experiment: {config_name}")
        print(f"{'='*80}")
        
        if dry_run:
            print(f"[DRY RUN] Would run: {train_script} experiment=ablation/{config_name}")
            return None
        
        # 构建训练命令
        cmd = [
            'python', train_script,
            f'experiment=ablation/{config_name}',
        ]
        
        # 执行训练
        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode != 0:
                print(f"Warning: Experiment {config_name} returned non-zero exit code")
                print(f"Error output: {result.stderr}")
                return None
            
            # 尝试加载结果
            exp_dir = f"./experiments/ablation/{config_name}"
            metrics_path = os.path.join(exp_dir, "final_metrics.json")
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                return {
                    'name': config_name,
                    'status': 'completed',
                    'metrics': metrics,
                    'path': exp_dir,
                }
            else:
                print(f"Warning: Metrics file not found: {metrics_path}")
                return {
                    'name': config_name,
                    'status': 'completed_no_metrics',
                    'path': exp_dir,
                }
        
        except Exception as e:
            print(f"Error running experiment {config_name}: {e}")
            return {
                'name': config_name,
                'status': 'failed',
                'error': str(e),
            }
    
    def run_all_experiments(
        self,
        experiments: Optional[List[str]] = None,
        dry_run: bool = False,
        parallel: bool = False,
        max_parallel: int = 2,
    ) -> Dict[str, Dict[str, Any]]:
        """
        运行所有消融实验
        
        Args:
            experiments: 要运行的实验列表，如果为None则运行所有
            dry_run: 是否为试运行
            parallel: 是否并行运行
            max_parallel: 最大并行数
        
        Returns:
            所有实验结果字典
        """
        configs = self.generate_ablation_configs()
        
        if experiments:
            configs = {k: v for k, v in configs.items() if k in experiments}
        
        results = {}
        
        if parallel and not dry_run:
            # 并行运行（简化版本，实际可以使用multiprocessing）
            print("Parallel execution not fully implemented, running sequentially")
        
        for config_name, config_path in configs.items():
            result = self.run_experiment(config_name, config_path, dry_run=dry_run)
            if result:
                results[config_name] = result
        
        return results
    
    def aggregate_results(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        汇总实验结果到对比表格
        
        Args:
            results: 实验结果字典
            output_path: 输出路径（CSV格式）
        
        Returns:
            pandas DataFrame包含对比结果
        """
        # 准备数据
        rows = []
        
        for exp_name, exp_result in results.items():
            row = {
                'Experiment': exp_name,
                'Status': exp_result.get('status', 'unknown'),
            }
            
            # 提取指标
            metrics = exp_result.get('metrics', {})
            best_metrics = metrics.get('best_metrics', {})
            final_metrics = metrics.get('final_metrics', {})
            
            # 添加所有指标
            all_metrics = {**best_metrics, **final_metrics}
            for metric_name, metric_value in all_metrics.items():
                row[metric_name] = metric_value
            
            rows.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 保存
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nComparison table saved to: {output_path}")
        
        # 同时生成HTML和Markdown格式
        if output_path:
            html_path = output_path.replace('.csv', '.html')
            df.to_html(html_path, index=False, escape=False, classes='table table-striped')
            print(f"HTML table saved to: {html_path}")
            
            md_path = output_path.replace('.csv', '.md')
            try:
                df.to_markdown(md_path, index=False)
                print(f"Markdown table saved to: {md_path}")
            except AttributeError:
                # Fallback if to_markdown is not available
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(df.to_string(index=False))
                print(f"Markdown table (text format) saved to: {md_path}")
        
        return df
    
    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, Any]],
        baseline_name: str = 'base',
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成对比报告
        
        Args:
            results: 实验结果字典
            baseline_name: 基线实验名称
            output_path: 输出路径
        
        Returns:
            报告Markdown字符串
        """
        report = f"""# Ablation Study Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiments Summary

Total experiments: {len(results)}
Completed: {sum(1 for r in results.values() if r.get('status') == 'completed')}
Failed: {sum(1 for r in results.values() if r.get('status') == 'failed')}

## Experiment Details

"""
        
        for exp_name, exp_result in results.items():
            status = exp_result.get('status', 'unknown')
            report += f"### {exp_name}\n\n"
            report += f"- **Status**: {status}\n"
            
            if 'metrics' in exp_result:
                metrics = exp_result['metrics']
                best_metrics = metrics.get('best_metrics', {})
                report += "- **Best Metrics**:\n"
                for metric, value in best_metrics.items():
                    report += f"  - {metric}: {value:.4f}\n"
            
            if 'error' in exp_result:
                report += f"- **Error**: {exp_result['error']}\n"
            
            report += "\n"
        
        # 对比表格
        df = self.aggregate_results(results)
        report += "## Comparison Table\n\n"
        try:
            report += df.to_markdown(index=False)
        except AttributeError:
            # Fallback if to_markdown is not available
            report += df.to_string(index=False)
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Comparison report saved to: {output_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Generator")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate', 'run', 'aggregate', 'all'],
        default='all',
        help='Mode: generate configs, run experiments, aggregate results, or all'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        default=None,
        help='Specific experiments to run (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (generate configs but do not run experiments)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments/ablation',
        help='Output directory for experiments'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='./config/experiment/ablation',
        help='Directory for ablation config files'
    )
    parser.add_argument(
        '--train-script',
        type=str,
        default='train_robust_grpo_hydra.py',
        help='Training script path'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='base',
        help='Baseline experiment name for comparison'
    )
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = AblationStudyGenerator(
        output_dir=args.output_dir,
        config_dir=args.config_dir,
    )
    
    if args.mode in ['generate', 'all']:
        print("Generating ablation experiment configurations...")
        configs = generator.generate_ablation_configs()
        print(f"\nGenerated {len(configs)} ablation configurations:")
        for name, path in configs.items():
            print(f"  - {name}: {path}")
    
    if args.mode in ['run', 'all']:
        print("\nRunning ablation experiments...")
        results = generator.run_all_experiments(
            experiments=args.experiments,
            dry_run=args.dry_run,
        )
        
        # 保存结果
        results_path = os.path.join(args.output_dir, 'ablation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_path}")
    
    if args.mode in ['aggregate', 'all']:
        # 加载结果
        results_path = os.path.join(args.output_dir, 'ablation_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        else:
            # 从实验目录加载
            comparator = ExperimentComparator(args.output_dir)
            experiments = comparator.load_experiments()
            results = {
                exp['name']: {
                    'name': exp['name'],
                    'status': 'completed',
                    'metrics': exp['metrics'],
                    'path': exp['path'],
                }
                for exp in experiments
            }
        
        print("\nAggregating results...")
        df = generator.aggregate_results(
            results,
            output_path=os.path.join(args.output_dir, 'ablation_comparison.csv'),
        )
        
        # 生成对比报告
        report = generator.generate_comparison_report(
            results,
            baseline_name=args.baseline,
            output_path=os.path.join(args.output_dir, 'ablation_report.md'),
        )
        
        print("\n" + "="*80)
        print("Ablation Study Complete!")
        print("="*80)
        print(f"\nResults summary:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

