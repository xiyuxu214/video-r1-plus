# Configuration Guide

This directory contains Hydra configuration files for Robust-T-GRPO training.

## Directory Structure

```
config/
├── model/              # Model architecture configurations
│   ├── robust_grpo.yaml
│   └── selection_policy.yaml
├── training/           # Training configurations
│   ├── rl_training.yaml
│   └── reward_weights.yaml
├── data/               # Data and augmentation configurations
│   ├── noise_schedule.yaml
│   └── dataset_mix.yaml
├── experiment/         # Experiment configurations
│   ├── base.yaml
│   └── ablation/      # Ablation study configurations
│       ├── no_selection_policy.yaml
│       ├── no_curriculum.yaml
│       ├── no_temporal.yaml
│       └── high_noise.yaml
└── config.yaml         # Main Hydra config file
```

## Usage

### Basic Training

```bash
# Use default configuration (base experiment)
python train_robust_grpo_hydra.py

# Use specific experiment configuration
python train_robust_grpo_hydra.py experiment=ablation/no_selection_policy

# Override specific parameters
python train_robust_grpo_hydra.py experiment=base model.name_or_path=Qwen/Qwen2-VL-7B training.learning_rate=2e-6

# Use different model configuration
python train_robust_grpo_hydra.py model=robust_grpo model/selection_policy=selection_policy

# Use different training configuration
python train_robust_grpo_hydra.py training=rl_training training/reward_weights=reward_weights
```

### Multi-run (Hyperparameter Sweep)

```bash
# Sweep over learning rates
python train_robust_grpo_hydra.py -m training.learning_rate=1e-6,2e-6,5e-6

# Sweep over multiple parameters
python train_robust_grpo_hydra.py -m training.learning_rate=1e-6,2e-6 training.beta=0.02,0.04,0.06

# Sweep over different experiments
python train_robust_grpo_hydra.py -m experiment=base,ablation/no_selection_policy,ablation/no_curriculum
```

### Configuration Groups

- `model`: Model architecture and initialization
- `training`: Training hyperparameters and strategies
- `data`: Dataset and augmentation settings
- `experiment`: Complete experiment configurations

## Configuration Files

### Model Configurations

- **robust_grpo.yaml**: Main model architecture configuration
- **selection_policy.yaml**: Selection augmented policy configuration

### Training Configurations

- **rl_training.yaml**: Reinforcement learning training parameters
- **reward_weights.yaml**: Reward component weights and configurations

### Data Configurations

- **noise_schedule.yaml**: Curriculum learning and noise augmentation
- **dataset_mix.yaml**: Dataset loading and mixing strategies

### Experiment Configurations

- **base.yaml**: Base experiment with all features enabled
- **ablation/**: Ablation study configurations

## Custom Configuration

To create a custom experiment configuration:

1. Create a new YAML file in `config/experiment/`
2. Use `defaults` to inherit from base configuration
3. Override specific parameters as needed

Example:

```yaml
# Custom Experiment Configuration

defaults:
  - ../base
  - _self_

experiment:
  name: "my-custom-experiment"
  description: "My custom experiment"
  output_dir: "./experiments/my-custom-experiment"
  
# Override specific parameters
training:
  learning_rate: 2e-6
  beta: 0.05
  num_train_epochs: 2

model:
  name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
```

## Notes

- All paths are relative to the project root
- Use `${variable}` syntax for variable substitution
- Configuration files support YAML comments
- Use `null` to explicitly set a value to None

