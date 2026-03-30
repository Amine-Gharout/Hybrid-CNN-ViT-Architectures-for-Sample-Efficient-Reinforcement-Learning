# Hybrid CNN-ViT Architectures for Sample-Efficient Reinforcement Learning

This repository contains the codebase for our systematic benchmark evaluating Vision Transformers (ViTs), standard CNNs, and Hybrid CNN-ViT architectures in low-data RL regimes. The implementations are designed for the MinAtar environment suite and evaluated under three algorithms: PPO, DQN, and Distributional C51.

## Key Features

- **Architectures**: Convolutional Neural Networks (CNN), Vision Transformers (ViT), Hybrid CNN-ViT with Gated Fusion, and multimodal variants (ViT + Text).
- **Algorithms**: Implementations of PPO (on-policy), Double Dueling DQN (off-policy), and Categorical C51 (distributional).
- **Efficiency**: Designed for constrained hardware. A full 144-run factorial benchmark (3 algos × 4 architectures × 3 games × 3 seeds) can be run on a single 8GB consumer GPU.
- **Environments**: Optimized for MinAtar (Breakout, Space Invaders, Freeway).

## Quick Start

### Installation

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

To run a single training session:
```bash
# Train PPO with Hybrid architecture on MinAtar Breakout
python algos/train_ppo.py --env=Breakout-v1 --condition=hybrid --seed=42
```

To run the full benchmark suite:
```bash
python run_full_benchmark.py --output-dir=results/full_benchmark
```

## Directory Structure

```text
.
├── algos/                # Core RL algorithms (DQN, PPO, C51)
├── configs/              # Hyperparameter configurations (YAML)
├── docs/                 # Project documentation, research proposals, and poster templates
├── envs/                 # Environment wrappers and MinAtar integrations
├── evaluation/           # Scripts for checkpoint evaluation and statistical plots
├── models/               # Neural network backbones (CNN, ViT, Gated Fusion, Text Encoders)
├── scripts/              # Supplementary test and benchmark utility scripts
├── utils/                # Logging and misc utilities
├── run_full_benchmark.py # Main benchmark entry point
└── requirements.txt      # Dependencies
```

## Available Conditions

- `cnn_only`: Standard convolutional backbone (default baseline).
- `vit_only`: purely Vision Transformer based feature extraction.
- `hybrid`: Combining CNN local features with ViT global context via a learnable gating mechanism.
- `vit_text_*`: Exploratory multimodal baselines using frozen language models (e.g. DistilBERT).

## Citation

If you use this benchmark or the hybrid architecture codebase, please attribute:
```text
Author: Karim Merzouk
Co-Author: Amine Gharout
```

## License

MIT License
