# Setup Guide for Windows

## ⚡ Quick Fix for ale-py Installation Error

**You're getting the error because Python 3.14 is too new - ale-py doesn't have pre-built wheels yet.**

**FASTEST SOLUTION (recommended):**

```bash
# 1. Download and install Python 3.11 from python.org
# 2. Create new environment:
py -3.11 -m venv vit_rl_env
vit_rl_env\Scripts\activate

# 3. Install everything:
pip install torch torchvision timm transformers gymnasium opencv-python scipy pyyaml tqdm matplotlib seaborn pandas tensorboard wandb
pip install ale-py
pip install "gymnasium[atari,accept-rom-license]"

# 4. Test:
python test_setup.py
```

**OR Test Everything Right Now Without Atari:**

```bash
# Your current setup works! Just test with CartPole:
python quickstart.py
```

---

## Quick Setup

✓ **Core dependencies installed and working!**

## For Full Atari Benchmarks

**Problem:** ale-py doesn't have pre-built wheels for Python 3.14 yet and requires C++ compilation.

**Solutions (choose ONE):**

### Option 1: Use Python 3.11 (RECOMMENDED - Easiest)

```bash
# Install Python 3.11 from python.org
# Create new environment with Python 3.11
py -3.11 -m venv vit_rl_env_311
vit_rl_env_311\Scripts\activate

# Install e4: Use WSL2 (Windows Subsystem for Linux)

```bash
# In WSL2 Ubuntu terminal:
pip install "gymnasium[atari,accept-rom-license]"
# This works perfectly - no C++ compilation issues on Linux
```

### Option 5: Start with CartPole (TEST EVERYTHING NOW
### Option 3: Install Visual Studio Build Tools (For Advanced Users)

```bash
# 1. Download Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/
# 
# 2. Install with "Desktop development with C++" workload
# 
# 3. Restart terminal and run:
pip install ale-py
```

### Option 2: Install Visual Studio Build Tools

1. Download **Visual Studio Build Tools** from: https://visualstudio.microsoft.com/downloads/
2. Install with **"Desktop development with C++"** workload
3. Then run:
```bash
pip install ale-py
pip install "gymnasium[atari]"
```

### Option 3: Use WSL2 (Windows Subsystem for Linux)

```bash
# In WSL2 Ubuntu
pip install "gymnasium[atari,accept-rom-license]"
```

### Option 4: Start with CartPole (For Testing)

You can test the entire pipeline with CartPole first:

```python
# Modify configs/experiments.yaml
environments:
  cartpole: "CartPole-v1"
  
# Then run
python algos/train_ppo.py --env=CartPole-v1 --condition=vit_only --seed=42
```

## Current Status

✓ All Python dependencies installed
✓ Core models working (ViT, Text Encoder, Fusion)
✓ Text history buffer working
✓ Training pipeline ready
✓ Evaluation scripts ready

⚠ Atari ROMs not installed (use one of the options above)
⚠ CUDA not available (CPU training will be slow but functional)

## GPU Setup (Optional but Recommended)

For faster training, install CUDA-enabled PyTorch:

```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install CUDA version (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Testing Your Setup

Run the test suite:
```bash
python test_setup.py
```

## Quick Start Without Atari

While waiting for Atari setup, you can run everything with CartPole:

```bash
# Test training
python algos/train_ppo.py --env=CartPole-v1 --condition=vit_only --seed=42 --output-dir=results/cartpole_test

# This will verify:
# - ViT model loads and trains
# - PPO algorithm works
# - Checkpointing works
# - Logging works
```

Once CartPole trains successfully, you'll know the entire pipeline works, and you just need to add Atari environments.
