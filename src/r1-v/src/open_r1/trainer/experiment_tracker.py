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
Experiment Tracker Module

Complete experiment tracking system with:
1. WandB/MLflow integration
2. Hyperparameter search support (Optuna)
3. Experiment comparison analysis
4. Model Card generation
5. Reproducibility checks (random seeds, environment versions)

Outputs standard experiment report templates.
"""

import os
import json
import sys
import platform
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
import subprocess
import pkg_resources

import torch
import numpy as np
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not available")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available")


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str
    project_name: str = "robust-t-grpo"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    git_commit: Optional[str] = None
    environment_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReproducibilityInfo:
    """可重复性信息"""
    random_seed: int
    python_version: str
    torch_version: str
    cuda_version: Optional[str] = None
    git_commit: Optional[str] = None
    git_diff: Optional[str] = None
    package_versions: Dict[str, str] = field(default_factory=dict)
    environment_hash: Optional[str] = None


class ExperimentTracker:
    """
    实验跟踪器
    
    集成WandB、MLflow、Optuna等功能。
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        use_wandb: bool = True,
        use_mlflow: bool = False,
        wandb_project: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        output_dir: str = "./experiments",
    ):
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置随机种子
        self._set_random_seeds(config.random_seed)
        
        # 收集可重复性信息
        self.reproducibility_info = self._collect_reproducibility_info()
        
        # 初始化WandB
        if self.use_wandb:
            wandb_project = wandb_project or config.project_name
            wandb.init(
                project=wandb_project,
                name=config.experiment_name,
                config=asdict(config),
                tags=config.tags,
                notes=config.description,
            )
            # 记录可重复性信息
            wandb.config.update(asdict(self.reproducibility_info))
        
        # 初始化MLflow
        if self.use_mlflow:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(config.project_name)
            mlflow.start_run(run_name=config.experiment_name)
            mlflow.log_params(config.hyperparameters)
            mlflow.set_tags({tag: "true" for tag in config.tags})
        
        # 实验指标
        self.metrics_history: Dict[str, List[float]] = {}
        self.best_metrics: Dict[str, float] = {}
        
        # 保存配置
        self._save_config()
    
    def _set_random_seeds(self, seed: int):
        """设置所有随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _collect_reproducibility_info(self) -> ReproducibilityInfo:
        """收集可重复性信息"""
        # Python版本
        python_version = sys.version
        
        # PyTorch版本
        torch_version = torch.__version__
        
        # CUDA版本
        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
        
        # Git信息
        git_commit = self._get_git_commit()
        git_diff = self._get_git_diff()
        
        # 包版本
        package_versions = self._get_package_versions()
        
        # 环境哈希
        env_str = f"{python_version}{torch_version}{cuda_version}{git_commit}"
        environment_hash = hashlib.md5(env_str.encode()).hexdigest()
        
        return ReproducibilityInfo(
            random_seed=self.config.random_seed,
            python_version=python_version,
            torch_version=torch_version,
            cuda_version=cuda_version,
            git_commit=git_commit,
            git_diff=git_diff,
            package_versions=package_versions,
            environment_hash=environment_hash,
        )
    
    def _get_git_commit(self) -> Optional[str]:
        """获取Git提交哈希"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _get_git_diff(self) -> Optional[str]:
        """获取Git差异"""
        try:
            result = subprocess.run(
                ['git', 'diff'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout if result.stdout else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _get_package_versions(self) -> Dict[str, str]:
        """获取关键包的版本"""
        packages = [
            'torch', 'transformers', 'trl', 'numpy', 'pandas',
            'matplotlib', 'seaborn', 'optuna', 'wandb', 'mlflow'
        ]
        versions = {}
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                versions[package] = version
            except pkg_resources.DistributionNotFound:
                pass
        return versions
    
    def _save_config(self):
        """保存配置"""
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_config": asdict(self.config),
                "reproducibility_info": asdict(self.reproducibility_info),
            }, f, indent=2, ensure_ascii=False)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标"""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
            
            # 更新最佳指标
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                # 假设越大越好（可以根据指标类型调整）
                if value > self.best_metrics[key]:
                    self.best_metrics[key] = value
        
        # 记录到WandB
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # 记录到MLflow
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """记录超参数"""
        self.config.hyperparameters.update(hyperparameters)
        
        if self.use_wandb:
            wandb.config.update(hyperparameters)
        
        if self.use_mlflow:
            mlflow.log_params(hyperparameters)
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """记录文件"""
        if self.use_wandb:
            wandb.log_artifact(file_path, type=artifact_type)
        
        if self.use_mlflow:
            mlflow.log_artifact(file_path)
    
    def save_model(self, model, model_name: str = "model"):
        """保存模型"""
        model_path = os.path.join(self.output_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, model_name)
        
        self.log_artifact(model_path, "model")
    
    def finish(self):
        """结束实验"""
        # 保存最终指标
        final_metrics_path = os.path.join(self.output_dir, "final_metrics.json")
        with open(final_metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                "best_metrics": self.best_metrics,
                "final_metrics": {k: v[-1] if v else 0.0 for k, v in self.metrics_history.items()},
            }, f, indent=2, ensure_ascii=False)
        
        if self.use_wandb:
            wandb.finish()
        
        if self.use_mlflow:
            mlflow.end_run()


class HyperparameterSearch:
    """
    超参数搜索（基于Optuna）
    """
    
    def __init__(
        self,
        study_name: str,
        direction: str = "maximize",  # "maximize" or "minimize"
        n_trials: int = 100,
        storage: Optional[str] = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter search")
        
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        
        # 创建或加载study
        if storage:
            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                storage=storage,
                load_if_exists=True,
            )
        else:
            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction,
            )
    
    def suggest_hyperparameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据搜索空间建议超参数
        
        Args:
            trial: Optuna trial对象
            search_space: 搜索空间定义，例如：
                {
                    'learning_rate': ('log_uniform', 1e-6, 1e-4),
                    'batch_size': ('int', 1, 16),
                    'beta': ('uniform', 0.01, 0.1),
                }
        
        Returns:
            建议的超参数字典
        """
        suggested = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, tuple):
                dist_type = param_config[0]
                if dist_type == 'log_uniform':
                    suggested[param_name] = trial.suggest_loguniform(param_name, param_config[1], param_config[2])
                elif dist_type == 'uniform':
                    suggested[param_name] = trial.suggest_uniform(param_name, param_config[1], param_config[2])
                elif dist_type == 'int':
                    suggested[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                elif dist_type == 'categorical':
                    suggested[param_name] = trial.suggest_categorical(param_name, param_config[1])
            else:
                # 直接使用值
                suggested[param_name] = param_config
        
        return suggested
    
    def optimize(
        self,
        objective_func: Callable[[optuna.Trial], float],
        n_trials: Optional[int] = None,
    ) -> optuna.Study:
        """
        执行超参数优化
        
        Args:
            objective_func: 目标函数，接受trial参数，返回要优化的值
            n_trials: 试验次数，如果为None则使用self.n_trials
        
        Returns:
            Study对象
        """
        n_trials = n_trials or self.n_trials
        self.study.optimize(objective_func, n_trials=n_trials)
        return self.study
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取最佳超参数"""
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """获取最佳值"""
        return self.study.best_value
    
    def plot_optimization_history(self, output_path: Optional[str] = None):
        """绘制优化历史"""
        if not OPTUNA_AVAILABLE:
            return
        
        try:
            fig = optuna.visualization.plot_optimization_history(self.study)
            if output_path:
                fig.write_html(output_path)
            else:
                fig.show()
        except Exception as e:
            print(f"Error plotting optimization history: {e}")


class ExperimentComparator:
    """
    实验对比分析器
    """
    
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = experiments_dir
    
    def load_experiments(self) -> List[Dict[str, Any]]:
        """加载所有实验"""
        experiments = []
        
        for exp_dir in Path(self.experiments_dir).iterdir():
            if exp_dir.is_dir():
                config_path = exp_dir / "config.json"
                metrics_path = exp_dir / "final_metrics.json"
                
                if config_path.exists() and metrics_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    
                    experiments.append({
                        "name": exp_dir.name,
                        "config": config,
                        "metrics": metrics,
                        "path": str(exp_dir),
                    })
        
        return experiments
    
    def compare_experiments(
        self,
        experiment_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对比实验
        
        Args:
            experiment_names: 要对比的实验名称列表，如果为None则对比所有
            output_path: 输出路径
        
        Returns:
            对比结果字典
        """
        experiments = self.load_experiments()
        
        if experiment_names:
            experiments = [e for e in experiments if e["name"] in experiment_names]
        
        comparison = {
            "experiments": [],
            "best_per_experiment": {},
            "summary": {},
        }
        
        all_metrics = set()
        for exp in experiments:
            exp_metrics = exp["metrics"].get("best_metrics", {})
            all_metrics.update(exp_metrics.keys())
            
            comparison["experiments"].append({
                "name": exp["name"],
                "metrics": exp_metrics,
                "config": exp["config"],
            })
        
        # 找出每个指标的最佳实验
        for metric in all_metrics:
            best_exp = None
            best_value = float('-inf')
            
            for exp in experiments:
                value = exp["metrics"].get("best_metrics", {}).get(metric, float('-inf'))
                if value > best_value:
                    best_value = value
                    best_exp = exp["name"]
            
            comparison["best_per_experiment"][metric] = {
                "experiment": best_exp,
                "value": best_value,
            }
        
        # 保存对比结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        return comparison


class ModelCardGenerator:
    """
    模型卡生成器
    """
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        self.tracker = experiment_tracker
    
    def generate(
        self,
        model_description: str = "",
        training_data: str = "",
        evaluation_results: Optional[Dict[str, float]] = None,
        limitations: List[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成模型卡
        
        Args:
            model_description: 模型描述
            training_data: 训练数据描述
            evaluation_results: 评估结果
            limitations: 模型限制列表
            output_path: 输出路径
        
        Returns:
            模型卡Markdown字符串
        """
        limitations = limitations or []
        evaluation_results = evaluation_results or self.tracker.best_metrics
        
        # 生成模型卡
        model_card = f"""# Model Card: {self.tracker.config.experiment_name}

## Model Details

### Model Description
{model_description or 'Robust-T-GRPO model for video understanding.'}

### Model Date
{datetime.now().strftime('%Y-%m-%d')}

### Model Version
1.0.0

## Training Details

### Training Data
{training_data or 'Video-R1 dataset'}

### Training Procedure
- **Framework**: Robust-T-GRPO
- **Random Seed**: {self.tracker.config.random_seed}
- **Hyperparameters**:
"""
        
        for key, value in self.tracker.config.hyperparameters.items():
            model_card += f"  - {key}: {value}\n"
        
        model_card += f"""
### Reproducibility
- **Python Version**: {self.tracker.reproducibility_info.python_version.split()[0]}
- **PyTorch Version**: {self.tracker.reproducibility_info.torch_version}
- **CUDA Version**: {self.tracker.reproducibility_info.cuda_version or 'N/A'}
- **Git Commit**: {self.tracker.reproducibility_info.git_commit or 'N/A'}
- **Environment Hash**: {self.tracker.reproducibility_info.environment_hash}

## Evaluation

### Metrics
"""
        
        for metric, value in evaluation_results.items():
            model_card += f"- **{metric}**: {value:.4f}\n"
        
        model_card += """
## Limitations

"""
        
        if limitations:
            for limitation in limitations:
                model_card += f"- {limitation}\n"
        else:
            model_card += "- Model performance may vary on different video domains.\n"
            model_card += "- Performance degrades with high noise levels.\n"
        
        model_card += """
## Citation

```bibtex
@article{{robust_t_grpo,
  title={{Robust Temporal Group Relative Policy Optimization}},
  author={{Your Name}},
  year=2025,
}}
```
"""
        
        # 保存模型卡
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            print(f"Model card saved to: {output_path}")
        
        return model_card


def generate_experiment_report(
    experiment_tracker: ExperimentTracker,
    output_path: Optional[str] = None,
) -> str:
    """
    生成标准实验报告
    
    Args:
        experiment_tracker: 实验跟踪器
        output_path: 输出路径
    
    Returns:
        报告Markdown字符串
    """
    report = f"""# Experiment Report: {experiment_tracker.config.experiment_name}

## Experiment Information

- **Project**: {experiment_tracker.config.project_name}
- **Description**: {experiment_tracker.config.description}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Tags**: {', '.join(experiment_tracker.config.tags)}

## Hyperparameters

"""
    
    for key, value in experiment_tracker.config.hyperparameters.items():
        report += f"- **{key}**: {value}\n"
    
    report += """
## Best Metrics

"""
    
    for metric, value in experiment_tracker.best_metrics.items():
        report += f"- **{metric}**: {value:.4f}\n"
    
    report += """
## Reproducibility

"""
    
    report += f"- **Random Seed**: {experiment_tracker.reproducibility_info.random_seed}\n"
    report += f"- **Python Version**: {experiment_tracker.reproducibility_info.python_version}\n"
    report += f"- **PyTorch Version**: {experiment_tracker.reproducibility_info.torch_version}\n"
    report += f"- **CUDA Version**: {experiment_tracker.reproducibility_info.cuda_version or 'N/A'}\n"
    report += f"- **Git Commit**: {experiment_tracker.reproducibility_info.git_commit or 'N/A'}\n"
    report += f"- **Environment Hash**: {experiment_tracker.reproducibility_info.environment_hash}\n"
    
    report += """
## Package Versions

"""
    
    for package, version in experiment_tracker.reproducibility_info.package_versions.items():
        report += f"- **{package}**: {version}\n"
    
    # 保存报告
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Experiment report saved to: {output_path}")
    
    return report


# 便捷函数
def create_experiment_tracker(
    experiment_name: str,
    project_name: str = "robust-t-grpo",
    use_wandb: bool = True,
    use_mlflow: bool = False,
    random_seed: int = 42,
    output_dir: str = "./experiments",
    **kwargs,
) -> ExperimentTracker:
    """
    创建实验跟踪器的便捷函数
    
    Args:
        experiment_name: 实验名称
        project_name: 项目名称
        use_wandb: 是否使用WandB
        use_mlflow: 是否使用MLflow
        random_seed: 随机种子
        output_dir: 输出目录
        **kwargs: 其他配置参数
    
    Returns:
        ExperimentTracker实例
    """
    config = ExperimentConfig(
        experiment_name=experiment_name,
        project_name=project_name,
        random_seed=random_seed,
        **kwargs,
    )
    
    return ExperimentTracker(
        config=config,
        use_wandb=use_wandb,
        use_mlflow=use_mlflow,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    print("Experiment tracker module loaded successfully!")
    print("Available classes:")
    print("  - ExperimentTracker")
    print("  - HyperparameterSearch")
    print("  - ExperimentComparator")
    print("  - ModelCardGenerator")

