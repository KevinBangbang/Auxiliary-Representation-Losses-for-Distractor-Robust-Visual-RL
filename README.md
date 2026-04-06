# Auxiliary Representation Losses for Distractor-Robust Visual RL

**Authors:** Bangcheng Wang, Kean Chen | **Course:** CSC415 Introduction to Reinforcement Learning

This repository extends [DrQ-v2](https://github.com/facebookresearch/drqv2) with two lightweight auxiliary representation losses to improve robustness under visual distractors.

## Modifications

### Mod A: Stop-Gradient Consistency Regularizer
A task-agnostic loss encouraging augmentation-invariant representations:
```
L_consistency = ||sg[enc(f1(o))] - enc(f2(o))||^2
```
- Zero computational overhead
- Robust to loss weight α across [0.01, 1.0]

### Mod B: Task-Conditional Contrastive Loss
An InfoNCE loss using Q-value proximity to define positive pairs:
```
(i,j) positive if |Q(si,ai) - Q(sj,aj)| < ε
```
- Task-aware: groups states by behavioral similarity
- +25% overhead per training step

### Distractor Environment
DMC-GB protocol: video backgrounds overlaid using MuJoCo depth-buffer segmentation.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install mujoco dm_control hydra-core==1.3.2 omegaconf==2.3.0 \
    numpy termcolor imageio imageio-ffmpeg tb-nightly pandas matplotlib \
    opencv-python-headless scikit-learn
```

**Requirements:** Python 3.12+, CUDA-capable GPU, mujoco 3.x, dm_control 1.x

## Quick Start

```bash
# Baseline (clean)
python train.py "task@_global_=cartpole_swingup"

# Mod A (consistency loss)
python train.py "task@_global_=cartpole_swingup" use_consistency=true

# Mod B (contrastive loss)
python train.py "task@_global_=cartpole_swingup" use_contrastive=true

# Both modifications
python train.py "task@_global_=cartpole_swingup" use_consistency=true use_contrastive=true

# With distractors
python train.py "task@_global_=cartpole_swingup" use_distractors=true use_consistency=true

# Mod B with warm-start (delay contrastive loss for 100K frames)
python train.py "task@_global_=cartpole_swingup" use_contrastive=true contrastive_warmstart_steps=100000
```

## Reproduce All Experiments

```bash
# Run all experiments (cartpole 3 seeds + walker/cheetah seed=1)
bash scripts/run_remaining.sh

# Alpha sensitivity sweep
bash scripts/run_alpha_sweep.sh

# Representation analysis (after training)
python scripts/eval_representations.py
python scripts/analyze_representations.py

# Generate figures
python scripts/plot_experiment_results.py
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_consistency` | `false` | Enable Mod A |
| `use_contrastive` | `false` | Enable Mod B |
| `use_distractors` | `false` | Enable video backgrounds |
| `consistency_alpha` | `0.1` | Mod A loss weight |
| `contrastive_alpha` | `0.1` | Mod B loss weight |
| `contrastive_tau` | `0.1` | Temperature for InfoNCE |
| `contrastive_epsilon` | `5.0` | Q-value threshold for positive pairs |
| `contrastive_warmstart_steps` | `0` | Delay contrastive loss activation (frames) |

## Repository Structure

```
drqv2/
├── train.py              # Main training script
├── drqv2.py              # Agent with Mod A & B
├── dmc.py                # DMC environment wrapper (distractor support)
├── cfgs/config.yaml      # Hydra configuration
├── scripts/
│   ├── run_remaining.sh          # Run all experiments
│   ├── run_alpha_sweep.sh        # Alpha sensitivity
│   ├── eval_representations.py   # Extract representations
│   ├── analyze_representations.py # CKA, effective rank, t-SNE
│   └── plot_experiment_results.py # Generate paper figures
├── figures/              # Generated plots
├── report.tex            # Final report (ICLR format)
└── references.bib        # Bibliography
```

## What's Changed (vs. Official DrQ-v2 Repo)

- **Python 3.12 + mujoco 3.x + dm_control 1.x** (no license required)
- **hydra-core 1.3.2** compatibility
- **Windows support** (skip `MUJOCO_GL=egl` on Windows)
- **Auxiliary losses** (Mod A: consistency, Mod B: contrastive)
- **Distractor environment** (video background overlay via depth segmentation)
- **Analysis scripts** (CKA, effective rank, t-SNE, plotting)

---

*Based on the [official DrQ-v2 codebase](https://github.com/facebookresearch/drqv2) by Yarats et al. (ICLR 2022). The majority of DrQ-v2 is licensed under the MIT license.*
