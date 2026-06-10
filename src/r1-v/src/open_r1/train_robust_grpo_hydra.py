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
Robust-T-GRPO Training Script with Hydra Configuration

This script uses Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os

from train_robust_grpo import main as train_main
from trl import GRPOConfig, ModelConfig
from trainer import RobustGRPOScriptArguments


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function with Hydra configuration
    
    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Convert Hydra config to training arguments
    # This is a simplified conversion - you may need to adjust based on your needs
    script_args = RobustGRPOScriptArguments(
        dataset_name=cfg.data.dataset.name,
        dataset_config=cfg.data.dataset.config,
        dataset_train_split=cfg.data.dataset.train_split,
        dataset_test_split=cfg.data.dataset.eval_split,
        reward_funcs=cfg.training.reward_functions.enabled,
        max_pixels=cfg.model.vision.max_pixels,
        min_pixels=cfg.model.vision.min_pixels,
        temporal=cfg.training.temporal.enabled,
        len_control=cfg.training.length_control.enabled,
        curriculum_learning=cfg.data.curriculum.enabled,
        curriculum_warmup_steps=cfg.data.curriculum.warmup_steps,
        curriculum_max_noise=cfg.data.curriculum.max_noise,
        use_selection_policy=cfg.model.selection_policy.enabled,
        selection_policy_weight=cfg.training.rewards.selection_weight,
        base_reward_weight=cfg.training.rewards.correctness_weight,
        selection_reward_weight=cfg.training.rewards.selection_weight,
        comparative_selection_reward_weight=0.15,  # Can be added to config
        robustness_reward_weight=cfg.training.rewards.robustness_penalty_weight,
    )
    
    # Create GRPOConfig from training config
    training_args = GRPOConfig(
        output_dir=cfg.experiment.output_dir,
        run_name=cfg.experiment.name,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.max_steps if cfg.training.max_steps > 0 else -1,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_steps=cfg.training.warmup_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        beta=cfg.training.beta,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps if cfg.training.eval_strategy == "steps" else None,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps if cfg.training.save_strategy == "steps" else None,
        save_total_limit=cfg.training.save_total_limit,
        logging_strategy=cfg.training.logging_strategy,
        logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to,
        seed=cfg.reproducibility.seed,
        max_prompt_length=cfg.model.generation.max_prompt_length,
        max_completion_length=cfg.model.generation.max_completion_length,
        num_generations=cfg.model.generation.num_generations,
    )
    
    # Create ModelConfig
    model_args = ModelConfig(
        model_name_or_path=cfg.model.name_or_path,
        model_revision=cfg.model.revision,
        torch_dtype=cfg.model.torch_dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    
    # Call main training function
    train_main(script_args, training_args, model_args)


if __name__ == "__main__":
    main()

