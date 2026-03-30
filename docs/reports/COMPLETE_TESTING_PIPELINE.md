# Complete Testing Pipeline & Implementation Guide
## Multimodal Vision Transformers for RL with Temporal Text Descriptions

**Project Goal:** Comprehensive benchmarking of multimodal (Vision + Temporal Text) RL agents vs. unimodal baselines

---

## Table of Contents
1. [Environment Setup & Dependencies](#1-environment-setup--dependencies)
2. [Implementation Roadmap](#2-implementation-roadmap)
3. [Detailed Implementation Files](#3-detailed-implementation-files)
4. [Comprehensive Benchmark Suite](#4-comprehensive-benchmark-suite)
5. [Experimental Protocol](#5-experimental-protocol)
6. [Results Analysis Pipeline](#6-results-analysis-pipeline)
7. [Timeline & Checklist](#7-timeline--checklist)

---

## 1. Environment Setup & Dependencies

### 1.1 System Requirements
```yaml
OS: Linux/Windows/macOS
Python: 3.9+
CUDA: 11.8+ (for GPU training)
RAM: 16GB minimum (32GB recommended)
Storage: 50GB for datasets, checkpoints, logs
```

### 1.2 Core Dependencies Installation

**File:** `requirements.txt`
```txt
# Core RL
gymnasium[atari]==0.29.1
stable-baselines3==2.2.1
torch==2.1.0
torchvision==0.16.0

# Vision Transformers
timm==0.9.12
transformers==4.36.0

# Text Processing
tokenizers==0.15.0
sentencepiece==0.1.99

# Utilities
numpy==1.24.3
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.0
wandb==0.16.1
tensorboard==2.15.1
pyyaml==6.0.1
tqdm==4.66.1
opencv-python==4.8.1.78
pillow==10.1.0

# Experiment Management
hydra-core==1.3.2
mlflow==2.9.2

# Testing & Profiling
pytest==7.4.3
pytest-cov==4.1.0
memory-profiler==0.61.0
line-profiler==4.1.1
```

### 1.3 Installation Commands
```bash
# Create virtual environment
python -m venv vit_rl_env
source vit_rl_env/bin/activate  # On Windows: vit_rl_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install ROMs for Atari (if needed)
pip install "gymnasium[accept-rom-license]"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
```

---

## 2. Implementation Roadmap

### Phase 1: Core Infrastructure (Days 1-3)
- [ ] Environment wrappers
- [ ] Base ViT encoder
- [ ] PPO baseline implementation
- [ ] Logging and checkpointing utilities

### Phase 2: Multimodal Components (Days 4-6)
- [ ] Text history buffer
- [ ] Text encoder module
- [ ] Fusion mechanisms (concat + cross-attention)
- [ ] Multimodal policy architecture

### Phase 3: Training Pipeline (Days 7-9)
- [ ] Unimodal training script
- [ ] Multimodal training script
- [ ] Hyperparameter configs
- [ ] Distributed training support

### Phase 4: Evaluation & Benchmarking (Days 10-12)
- [ ] Evaluation harness
- [ ] Ablation study setup
- [ ] Robustness tests
- [ ] Statistical analysis scripts

### Phase 5: Results & Visualization (Days 13-14)
- [ ] Plot generation
- [ ] Performance tables
- [ ] Poster-ready visualizations

---

## 3. Detailed Implementation Files

### 3.1 Environment Wrapper

**File:** `envs/atari_env.py`
```python
"""
Atari environment wrapper with standardized preprocessing
"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    RecordEpisodeStatistics,
)
from typing import Tuple, Optional
import cv2


class AtariEnvWrapper:
    """Wrapper for Atari environments with RL-standard preprocessing"""
    
    def __init__(
        self,
        env_name: str = "PongNoFrameskip-v4",
        frame_size: Tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        seed: Optional[int] = None,
        grayscale: bool = False,
        vit_mode: bool = True,  # If True, use RGB and resize to 224x224
    ):
        """
        Args:
            env_name: Name of Atari environment
            frame_size: Output frame dimensions
            frame_stack: Number of frames to stack
            seed: Random seed
            grayscale: Convert to grayscale
            vit_mode: Use ViT-friendly preprocessing (RGB, 224x224)
        """
        self.env_name = env_name
        self.vit_mode = vit_mode
        
        # Create base environment
        self.env = gym.make(env_name, render_mode=None)
        
        # Set seed
        if seed is not None:
            self.env.reset(seed=seed)
            np.random.seed(seed)
        
        # Apply preprocessing
        if vit_mode:
            # ViT expects 224x224 RGB images
            self.env = ViTAtariPreprocessing(
                self.env,
                frame_size=(224, 224),
                grayscale_obs=False,
            )
        else:
            # Standard CNN preprocessing
            self.env = AtariPreprocessing(
                self.env,
                frame_skip=1,
                screen_size=frame_size[0],
                grayscale_obs=grayscale,
                scale_obs=True,
            )
        
        # Frame stacking for temporal context
        if frame_stack > 1:
            self.env = FrameStack(self.env, num_stack=frame_stack)
        
        # Episode statistics
        self.env = RecordEpisodeStatistics(self.env)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped


class ViTAtariPreprocessing(gym.ObservationWrapper):
    """Custom preprocessing for ViT-based agents"""
    
    def __init__(
        self,
        env: gym.Env,
        frame_size: Tuple[int, int] = (224, 224),
        grayscale_obs: bool = False,
        normalize: bool = True,
    ):
        super().__init__(env)
        self.frame_size = frame_size
        self.grayscale_obs = grayscale_obs
        self.normalize = normalize
        
        # Update observation space
        if grayscale_obs:
            obs_shape = (*frame_size, 1)
        else:
            obs_shape = (*frame_size, 3)
        
        self.observation_space = gym.spaces.Box(
            low=0 if not normalize else 0.0,
            high=255 if not normalize else 1.0,
            shape=obs_shape,
            dtype=np.float32 if normalize else np.uint8,
        )
    
    def observation(self, obs):
        """Preprocess observation"""
        # Resize
        obs = cv2.resize(obs, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if self.grayscale_obs and len(obs.shape) == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=-1)
        
        # Normalize to [0, 1]
        if self.normalize:
            obs = obs.astype(np.float32) / 255.0
        
        return obs


def make_parallel_envs(
    env_name: str,
    num_envs: int = 8,
    seed: int = 42,
    vit_mode: bool = True,
):
    """Create vectorized parallel environments"""
    from gymnasium.vector import AsyncVectorEnv
    
    def make_env(rank: int):
        def _init():
            env = AtariEnvWrapper(
                env_name=env_name,
                seed=seed + rank,
                vit_mode=vit_mode,
            )
            return env.env
        return _init
    
    envs = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    return envs
```

---

### 3.2 Vision Transformer Backbone

**File:** `models/vit_policy.py`
```python
"""
Vision Transformer-based policy for RL
"""
import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional


class ViTBackbone(nn.Module):
    """Vision Transformer encoder for RL observations"""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_layers: int = 8,  # Freeze first N layers
        embedding_dim: int = 768,
    ):
        """
        Args:
            model_name: ViT model variant from timm
            pretrained: Use ImageNet pretrained weights
            freeze_layers: Number of transformer blocks to freeze
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        self.embedding_dim = embedding_dim
        
        # Freeze early layers for stability
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Get ViT output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            vit_output_dim = self.vit(dummy_input).shape[-1]
        
        # Projection to desired embedding dimension
        if vit_output_dim != embedding_dim:
            self.projection = nn.Linear(vit_output_dim, embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer blocks"""
        # Freeze patch embedding and cls token
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        
        self.vit.cls_token.requires_grad = False
        self.vit.pos_embed.requires_grad = False
        
        # Freeze transformer blocks
        if hasattr(self.vit, 'blocks'):
            for i, block in enumerate(self.vit.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor [B, C, H, W] or [B, F, C, H, W] (with frame stack)
        
        Returns:
            Embedding [B, embedding_dim]
        """
        # Handle frame stacking: take last frame
        if x.dim() == 5:  # [B, frames, C, H, W]
            x = x[:, -1]  # Take most recent frame
        
        # Ensure correct shape [B, 3, 224, 224]
        if x.shape[-2:] != (224, 224):
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        
        # Extract features
        features = self.vit(x)  # [B, vit_dim]
        
        # Project to embedding dimension
        embedding = self.projection(features)  # [B, embedding_dim]
        
        return embedding


class ActorCriticViT(nn.Module):
    """Actor-Critic policy using Vision Transformer"""
    
    def __init__(
        self,
        num_actions: int,
        vit_model: str = "vit_base_patch16_224",
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_vit_layers: int = 8,
    ):
        """
        Args:
            num_actions: Number of discrete actions
            vit_model: ViT model variant
            embedding_dim: ViT output embedding dimension
            hidden_dim: Hidden layer dimension for policy/value heads
            pretrained: Use pretrained weights
            freeze_vit_layers: Number of ViT layers to freeze
        """
        super().__init__()
        
        # Vision encoder
        self.encoder = ViTBackbone(
            model_name=vit_model,
            pretrained=pretrained,
            freeze_layers=freeze_vit_layers,
            embedding_dim=embedding_dim,
        )
        
        # Policy head (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Observation tensor [B, C, H, W]
        
        Returns:
            logits: Action logits [B, num_actions]
            value: State value [B, 1]
        """
        # Encode observation
        features = self.encoder(obs)
        
        # Get policy and value
        logits = self.policy_net(features)
        value = self.value_net(features)
        
        return logits, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate"""
        features = self.encoder(obs)
        return self.value_net(features)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value
        
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value
        """
        logits, value = self.forward(obs)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Get log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
```

---

### 3.3 Text History Buffer

**File:** `utils/text_history.py`
```python
"""
Text history buffer for temporal descriptions
"""
import numpy as np
from typing import List, Optional, Dict
from collections import deque


class HistoryBuffer:
    """Buffer to store and generate textual summaries of recent experience"""
    
    def __init__(
        self,
        history_length: int = 5,
        action_names: Optional[List[str]] = None,
        max_text_length: int = 512,
        template_style: str = "detailed",  # "detailed", "compact", "narrative"
    ):
        """
        Args:
            history_length: Number of recent steps to include
            action_names: Human-readable action names
            max_text_length: Maximum character length
            template_style: Style of text generation
        """
        self.history_length = history_length
        self.action_names = action_names
        self.max_text_length = max_text_length
        self.template_style = template_style
        
        # Circular buffer for history
        self.buffer = deque(maxlen=history_length)
        
        # Cumulative statistics
        self.total_reward = 0.0
        self.step_count = 0
    
    def add(self, action: int, reward: float, done: bool, info: Dict = None):
        """Add a transition to the buffer"""
        self.buffer.append({
            'action': action,
            'reward': reward,
            'done': done,
            'step': self.step_count,
            'info': info or {}
        })
        
        self.total_reward += reward
        self.step_count += 1
    
    def reset(self):
        """Reset buffer (call at episode start)"""
        self.buffer.clear()
        self.total_reward = 0.0
        self.step_count = 0
    
    def to_text(self) -> str:
        """Generate textual summary of history"""
        if len(self.buffer) == 0:
            return "Episode start. No previous actions."
        
        if self.template_style == "detailed":
            text = self._generate_detailed_text()
        elif self.template_style == "compact":
            text = self._generate_compact_text()
        elif self.template_style == "narrative":
            text = self._generate_narrative_text()
        else:
            text = self._generate_detailed_text()
        
        # Truncate if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return text
    
    def _generate_detailed_text(self) -> str:
        """Detailed step-by-step description"""
        parts = [f"Recent history (last {len(self.buffer)} steps):"]
        
        for i, transition in enumerate(self.buffer):
            action_str = self._get_action_name(transition['action'])
            reward_str = self._format_reward(transition['reward'])
            
            step_desc = f"Step {i+1}: Action={action_str}, Reward={reward_str}"
            
            if transition['done']:
                step_desc += " (Episode ended)"
            
            parts.append(step_desc)
        
        parts.append(f"Total reward so far: {self.total_reward:.2f}")
        
        return ". ".join(parts) + "."
    
    def _generate_compact_text(self) -> str:
        """Compact comma-separated format"""
        transitions = []
        
        for t in self.buffer:
            action = self._get_action_name(t['action'])
            reward = self._format_reward(t['reward'])
            transitions.append(f"({action},{reward})")
        
        return "History: " + " -> ".join(transitions)
    
    def _generate_narrative_text(self) -> str:
        """More natural language narrative"""
        if len(self.buffer) == 0:
            return "The agent has just started."
        
        # Analyze recent performance
        recent_rewards = [t['reward'] for t in self.buffer]
        avg_reward = np.mean(recent_rewards)
        
        # Build narrative
        narrative = []
        
        if avg_reward > 0.5:
            narrative.append("The agent is performing well recently.")
        elif avg_reward < -0.5:
            narrative.append("The agent has struggled in recent steps.")
        else:
            narrative.append("The agent's recent performance is mixed.")
        
        # Describe recent actions
        recent_actions = [self._get_action_name(t['action']) for t in list(self.buffer)[-3:]]
        narrative.append(f"Recent actions: {', '.join(recent_actions)}.")
        
        # Mention total reward
        narrative.append(f"Cumulative reward: {self.total_reward:.1f}.")
        
        return " ".join(narrative)
    
    def _get_action_name(self, action: int) -> str:
        """Convert action index to name"""
        if self.action_names and action < len(self.action_names):
            return self.action_names[action]
        return f"A{action}"
    
    def _format_reward(self, reward: float) -> str:
        """Format reward value"""
        if reward > 0:
            return f"+{reward:.1f}"
        elif reward < 0:
            return f"{reward:.1f}"
        else:
            return "0"


class ParallelHistoryBuffer:
    """Manage history buffers for multiple parallel environments"""
    
    def __init__(
        self,
        num_envs: int,
        history_length: int = 5,
        action_names: Optional[List[str]] = None,
        template_style: str = "detailed",
    ):
        """
        Args:
            num_envs: Number of parallel environments
            history_length: History length per environment
            action_names: Action names
            template_style: Text generation style
        """
        self.num_envs = num_envs
        self.buffers = [
            HistoryBuffer(
                history_length=history_length,
                action_names=action_names,
                template_style=template_style,
            )
            for _ in range(num_envs)
        ]
    
    def add(self, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, infos: List[Dict]):
        """Add transitions for all environments"""
        for i in range(self.num_envs):
            self.buffers[i].add(
                action=int(actions[i]),
                reward=float(rewards[i]),
                done=bool(dones[i]),
                info=infos[i] if i < len(infos) else {}
            )
            
            # Reset buffer if episode ended
            if dones[i]:
                self.buffers[i].reset()
    
    def to_text_batch(self) -> List[str]:
        """Get text summaries for all environments"""
        return [buffer.to_text() for buffer in self.buffers]
    
    def reset_all(self):
        """Reset all buffers"""
        for buffer in self.buffers:
            buffer.reset()
```

---

### 3.4 Text Encoder

**File:** `models/text_encoder.py`
```python
"""
Text encoder for temporal descriptions
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Union


class TextEncoder(nn.Module):
    """Text encoder using pretrained language model"""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 512,
        freeze: bool = True,
        max_length: int = 128,
    ):
        """
        Args:
            model_name: HuggingFace model name
            embedding_dim: Output embedding dimension
            freeze: Freeze pretrained weights
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get model output dimension
        hidden_size = self.model.config.hidden_size
        
        # Projection layer
        self.projection = nn.Linear(hidden_size, embedding_dim)
        
        # Freeze weights if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Args:
            texts: Text or batch of texts
        
        Returns:
            Embeddings [B, embedding_dim]
        """
        # Ensure batch format
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**encoded)
            # Use [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        
        # Project to desired dimension
        embedding = self.projection(cls_embedding)  # [B, embedding_dim]
        
        return embedding


class TextEncoderCache:
    """Cache for text encodings to avoid redundant computation"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
    
    def get(self, text: str, encoder: TextEncoder) -> torch.Tensor:
        """Get encoding from cache or compute"""
        if text not in self.cache:
            with torch.no_grad():
                encoding = encoder(text)
            
            # Add to cache
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[text] = encoding
        
        return self.cache[text]
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
```

---

### 3.5 Fusion Module

**File:** `models/fusion.py`
```python
"""
Multimodal fusion mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConcatFusion(nn.Module):
    """Simple concatenation-based fusion"""
    
    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
    ):
        """
        Args:
            vision_dim: Vision embedding dimension
            text_dim: Text embedding dimension
            output_dim: Fused output dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Fusion MLP
        self.fusion_net = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [B, vision_dim]
            text_features: [B, text_dim]
        
        Returns:
            fused_features: [B, output_dim]
        """
        # Concatenate
        combined = torch.cat([vision_features, text_features], dim=-1)
        
        # Fuse
        fused = self.fusion_net(combined)
        
        return fused


class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion"""
    
    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 512,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            vision_dim: Vision embedding dimension
            text_dim: Text embedding dimension
            output_dim: Output dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [B, vision_dim]
            text_features: [B, text_dim]
        
        Returns:
            fused_features: [B, output_dim]
        """
        batch_size = vision_features.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [B, output_dim]
        text_proj = self.text_proj(text_features)  # [B, output_dim]
        
        # Add sequence dimension for attention
        vision_seq = vision_proj.unsqueeze(1)  # [B, 1, output_dim]
        text_seq = text_proj.unsqueeze(1)  # [B, 1, output_dim]
        
        # Cross-attention: vision attends to text
        attn_output, _ = self.cross_attention(
            query=vision_seq,
            key=text_seq,
            value=text_seq,
        )  # [B, 1, output_dim]
        
        # Residual connection and normalization
        fused = self.norm1(vision_proj + self.dropout(attn_output.squeeze(1)))
        
        # Feed-forward network
        ffn_output = self.ffn(fused)
        fused = self.norm2(fused + self.dropout(ffn_output))
        
        return fused


class GatedFusion(nn.Module):
    """Gated fusion with learned modality importance"""
    
    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 512,
        output_dim: int = 512,
    ):
        """
        Args:
            vision_dim: Vision embedding dimension
            text_dim: Text embedding dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(vision_dim + text_dim, output_dim),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [B, vision_dim]
            text_features: [B, text_dim]
        
        Returns:
            fused_features: [B, output_dim]
        """
        # Project modalities
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # Compute gate
        gate_input = torch.cat([vision_features, text_features], dim=-1)
        gate_weight = self.gate(gate_input)  # [B, output_dim]
        
        # Weighted combination
        fused = gate_weight * vision_proj + (1 - gate_weight) * text_proj
        
        # Output projection
        output = self.output_proj(fused)
        
        return output
```

---

### 3.6 Multimodal Policy

**File:** `models/vit_text_policy.py`
```python
"""
Multimodal policy combining Vision Transformer and Text Encoder
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from models.vit_policy import ViTBackbone
from models.text_encoder import TextEncoder
from models.fusion import ConcatFusion, CrossAttentionFusion, GatedFusion


class ActorCriticViTText(nn.Module):
    """Multimodal Actor-Critic with Vision and Text"""
    
    def __init__(
        self,
        num_actions: int,
        vit_model: str = "vit_base_patch16_224",
        text_model: str = "distilbert-base-uncased",
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        fusion_type: str = "concat",  # "concat", "cross_attention", "gated"
        pretrained: bool = True,
        freeze_vit_layers: int = 8,
        freeze_text: bool = True,
    ):
        """
        Args:
            num_actions: Number of discrete actions
            vit_model: ViT model variant
            text_model: Text encoder model
            embedding_dim: Common embedding dimension
            hidden_dim: Hidden dimension for policy/value heads
            fusion_type: Type of fusion mechanism
            pretrained: Use pretrained weights
            freeze_vit_layers: Number of ViT layers to freeze
            freeze_text: Freeze text encoder
        """
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = ViTBackbone(
            model_name=vit_model,
            pretrained=pretrained,
            freeze_layers=freeze_vit_layers,
            embedding_dim=embedding_dim,
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_model,
            embedding_dim=embedding_dim,
            freeze=freeze_text,
        )
        
        # Fusion module
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                vision_dim=embedding_dim,
                text_dim=embedding_dim,
                output_dim=embedding_dim,
            )
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                vision_dim=embedding_dim,
                text_dim=embedding_dim,
                output_dim=embedding_dim,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                vision_dim=embedding_dim,
                text_dim=embedding_dim,
                output_dim=embedding_dim,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def encode(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
    ) -> torch.Tensor:
        """
        Encode multimodal observations
        
        Args:
            obs_vision: Visual observations [B, C, H, W]
            obs_text: Text descriptions (list of strings)
        
        Returns:
            Fused embedding [B, embedding_dim]
        """
        # Encode vision
        vision_features = self.vision_encoder(obs_vision)
        
        # Encode text
        text_features = self.text_encoder(obs_text)
        
        # Fuse modalities
        fused_features = self.fusion(vision_features, text_features)
        
        return fused_features
    
    def forward(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs_vision: Visual observations
            obs_text: Text descriptions
        
        Returns:
            logits: Action logits [B, num_actions]
            value: State values [B, 1]
        """
        # Get fused features
        features = self.encode(obs_vision, obs_text)
        
        # Compute policy and value
        logits = self.policy_net(features)
        value = self.value_net(features)
        
        return logits, value
    
    def get_value(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
    ) -> torch.Tensor:
        """Get value estimate"""
        features = self.encode(obs_vision, obs_text)
        return self.value_net(features)
    
    def get_action_and_value(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, and value
        
        Returns:
            action: Sampled or provided action
            log_prob: Log probability
            entropy: Policy entropy
            value: State value
        """
        logits, value = self.forward(obs_vision, obs_text)
        
        # Create distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        # Sample or use provided action
        if action is None:
            action = dist.sample()
        
        # Compute log prob and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
```

---

## 4. Comprehensive Benchmark Suite

### 4.1 Benchmark Environments

**Selected Environments:**
```python
BENCHMARK_ENVS = {
    # Tier 1: Primary benchmarks
    "pong": "PongNoFrameskip-v4",  # Simple, fast convergence
    "breakout": "BreakoutNoFrameskip-v4",  # Moderate complexity
    
    # Tier 2: Extended benchmarks
    "space_invaders": "SpaceInvadersNoFrameskip-v4",  # Rich visual features
    "ms_pacman": "MsPacmanNoFrameskip-v4",  # Temporal dependencies
    
    # Tier 3: Challenge benchmarks
    "montezuma_revenge": "MontezumaRevengeNoFrameskip-v4",  # Hard exploration
}
```

### 4.2 Experimental Conditions

**File:** `configs/experiments.yaml`
```yaml
# Baseline conditions
conditions:
  # 1. Vision-only (ViT)
  vit_only:
    name: "ViT-Only"
    vision_encoder: "vit_base_patch16_224"
    use_text: false
    pretrained: true
    freeze_vit_layers: 8
    
  # 2. Vision-only (CNN baseline)
  cnn_baseline:
    name: "CNN-Baseline"
    vision_encoder: "cnn"
    use_text: false
    
  # 3. ViT + Random Text (control)
  vit_random_text:
    name: "ViT+RandomText"
    vision_encoder: "vit_base_patch16_224"
    use_text: true
    text_random: true  # Shuffle or use random history
    fusion_type: "concat"
    
  # 4. ViT + True Temporal Text (main method)
  vit_text_detailed:
    name: "ViT+Text-Detailed"
    vision_encoder: "vit_base_patch16_224"
    use_text: true
    text_random: false
    text_style: "detailed"
    fusion_type: "concat"
    history_length: 5
    
  # 5. ViT + Text with Cross-Attention
  vit_text_crossattn:
    name: "ViT+Text-CrossAttn"
    vision_encoder: "vit_base_patch16_224"
    use_text: true
    fusion_type: "cross_attention"
    text_style: "detailed"
    history_length: 5
    
  # 6. ViT + Text with Gated Fusion
  vit_text_gated:
    name: "ViT+Text-Gated"
    vision_encoder: "vit_base_patch16_224"
    use_text: true
    fusion_type: "gated"
    text_style: "detailed"
    history_length: 5
    
  # Ablations: Different text styles
  vit_text_compact:
    name: "ViT+Text-Compact"
    use_text: true
    text_style: "compact"
    fusion_type: "concat"
    
  vit_text_narrative:
    name: "ViT+Text-Narrative"
    use_text: true
    text_style: "narrative"
    fusion_type: "concat"
    
  # Ablations: Different history lengths
  vit_text_short:
    name: "ViT+Text-H3"
    use_text: true
    history_length: 3
    
  vit_text_long:
    name: "ViT+Text-H10"
    use_text: true
    history_length: 10

# Training configuration
training:
  total_timesteps: 10_000_000  # 10M steps
  num_envs: 8  # Parallel environments
  num_steps: 128  # Rollout length
  num_epochs: 4  # PPO epochs per update
  batch_size: 256
  minibatch_size: 64
  
  # Optimizer
  learning_rate: 3.0e-4
  lr_schedule: "linear"  # Linear decay
  adam_epsilon: 1.0e-5
  max_grad_norm: 0.5
  
  # PPO hyperparameters
  gamma: 0.99  # Discount factor
  gae_lambda: 0.95  # GAE parameter
  clip_coef: 0.1  # PPO clip coefficient
  ent_coef: 0.01  # Entropy bonus
  vf_coef: 0.5  # Value loss coefficient
  
  # Logging
  log_interval: 10  # Log every N updates
  save_interval: 100  # Save checkpoint every N updates
  eval_interval: 50  # Evaluate every N updates
  eval_episodes: 10  # Number of evaluation episodes

# Seeds for reproducibility
seeds: [42, 123, 456, 789, 1024]

# Robustness tests
robustness_tests:
  # Partial observability
  frame_skip:
    skip_every: 3  # Drop every 3rd frame
    
  occlusion:
    type: "random_patches"  # Random black patches
    num_patches: 5
    patch_size: 32
    
  noise:
    type: "gaussian"
    std: 0.1
    
  # Visual distractors
  distractors:
    type: "moving_shapes"
    num_shapes: 3
```

### 4.3 Metrics Collection

**File:** `evaluation/metrics.py`
```python
"""
Comprehensive metrics collection
"""
import numpy as np
from typing import Dict, List
import json


class MetricsCollector:
    """Collect and aggregate training/evaluation metrics"""
    
    def __init__(self):
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'value_loss': [],
            'policy_loss': [],
            'entropy': [],
            'learning_rate': [],
            'timesteps': [],
        }
        
        self.eval_metrics = {
            'eval_returns': [],
            'eval_lengths': [],
            'eval_timesteps': [],
        }
    
    def add_training_metrics(self, step: int, metrics: Dict):
        """Add training metrics"""
        self.metrics['timesteps'].append(step)
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def add_eval_metrics(self, step: int, returns: List[float], lengths: List[int]):
        """Add evaluation metrics"""
        self.eval_metrics['eval_timesteps'].append(step)
        self.eval_metrics['eval_returns'].append(returns)
        self.eval_metrics['eval_lengths'].append(lengths)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {}
        
        # Training metrics
        if self.metrics['episode_returns']:
            returns = self.metrics['episode_returns']
            summary['final_train_return_mean'] = np.mean(returns[-100:])
            summary['final_train_return_std'] = np.std(returns[-100:])
        
        # Evaluation metrics
        if self.eval_metrics['eval_returns']:
            all_eval_returns = [r for episode_returns in self.eval_metrics['eval_returns'] 
                               for r in episode_returns]
            summary['final_eval_return_mean'] = np.mean(all_eval_returns[-100:])
            summary['final_eval_return_std'] = np.std(all_eval_returns[-100:])
            summary['max_eval_return'] = np.max(all_eval_returns)
        
        return summary
    
    def save(self, path: str):
        """Save metrics to file"""
        data = {
            'metrics': self.metrics,
            'eval_metrics': self.eval_metrics,
            'summary': self.get_summary(),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.metrics = data['metrics']
        self.eval_metrics = data['eval_metrics']


def compute_sample_efficiency(
    returns: List[float],
    timesteps: List[int],
    threshold: float = 10.0,
) -> int:
    """
    Compute steps needed to reach performance threshold
    
    Returns:
        Steps to threshold (or -1 if not reached)
    """
    for i, (ret, step) in enumerate(zip(returns, timesteps)):
        if ret >= threshold:
            return step
    return -1


def compute_auc(
    returns: List[float],
    timesteps: List[int],
    normalize: bool = True,
) -> float:
    """Compute area under the learning curve"""
    if len(returns) < 2:
        return 0.0
    
    auc = np.trapz(returns, timesteps)
    
    if normalize:
        auc = auc / (timesteps[-1] - timesteps[0])
    
    return auc
```

---

## 5. Experimental Protocol

### 5.1 Training Protocol

**File:** `scripts/run_full_benchmark.py`
```python
"""
Full benchmark execution script
"""
import argparse
import itertools
import subprocess
import yaml
from pathlib import Path


def run_experiment(
    env_name: str,
    condition: str,
    seed: int,
    config_path: str,
    output_dir: str,
):
    """Run a single experiment"""
    exp_name = f"{env_name}_{condition}_seed{seed}"
    
    cmd = [
        "python", "algos/train_ppo.py",
        f"--env={env_name}",
        f"--condition={condition}",
        f"--seed={seed}",
        f"--config={config_path}",
        f"--output-dir={output_dir}/{exp_name}",
        f"--wandb-project=vit-text-rl",
        f"--wandb-name={exp_name}",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel runs")
    parser.add_argument("--envs", nargs="+", default=None, help="Specific envs to run")
    parser.add_argument("--conditions", nargs="+", default=None, help="Specific conditions to run")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get environments
    envs = args.envs or list(BENCHMARK_ENVS.keys())
    
    # Get conditions
    conditions = args.conditions or list(config['conditions'].keys())
    
    # Get seeds
    seeds = config['seeds']
    
    # Generate all combinations
    experiments = list(itertools.product(envs, conditions, seeds))
    
    print(f"Total experiments: {len(experiments)}")
    print(f"Environments: {envs}")
    print(f"Conditions: {conditions}")
    print(f"Seeds: {seeds}")
    
    # Run experiments
    for env_name, condition, seed in experiments:
        try:
            run_experiment(
                env_name=BENCHMARK_ENVS[env_name],
                condition=condition,
                seed=seed,
                config_path=args.config,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"ERROR in {env_name}/{condition}/seed{seed}: {e}")
            continue


if __name__ == "__main__":
    main()
```

### 5.2 Statistical Analysis

**File:** `evaluation/statistical_analysis.py`
```python
"""
Statistical analysis of results
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple


def compare_conditions(
    baseline_results: List[float],
    treatment_results: List[float],
    test: str = "mannwhitneyu",
    alpha: float = 0.05,
) -> Dict:
    """
    Statistical comparison between two conditions
    
    Args:
        baseline_results: Results for baseline condition
        treatment_results: Results for treatment condition
        test: Statistical test ("mannwhitneyu", "ttest", "wilcoxon")
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    baseline = np.array(baseline_results)
    treatment = np.array(treatment_results)
    
    # Descriptive statistics
    stats_dict = {
        'baseline_mean': np.mean(baseline),
        'baseline_std': np.std(baseline),
        'baseline_median': np.median(baseline),
        'treatment_mean': np.mean(treatment),
        'treatment_std': np.std(treatment),
        'treatment_median': np.median(treatment),
        'mean_diff': np.mean(treatment) - np.mean(baseline),
        'median_diff': np.median(treatment) - np.median(baseline),
    }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(baseline)**2 + np.std(treatment)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(treatment) - np.mean(baseline)) / pooled_std
    else:
        cohens_d = 0.0
    stats_dict['cohens_d'] = cohens_d
    
    # Statistical test
    if test == "mannwhitneyu":
        statistic, p_value = stats.mannwhitneyu(treatment, baseline, alternative='two-sided')
    elif test == "ttest":
        statistic, p_value = stats.ttest_ind(treatment, baseline)
    elif test == "wilcoxon":
        statistic, p_value = stats.wilcoxon(treatment, baseline)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    stats_dict['test'] = test
    stats_dict['statistic'] = statistic
    stats_dict['p_value'] = p_value
    stats_dict['significant'] = p_value < alpha
    
    return stats_dict


def bootstrap_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval
    
    Returns:
        point_estimate, lower_bound, upper_bound
    """
    # Compute point estimate
    if statistic == "mean":
        point_est = np.mean(data)
        stat_func = np.mean
    elif statistic == "median":
        point_est = np.median(data)
        stat_func = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Bootstrap
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return point_est, lower, upper


def create_comparison_table(
    results_dict: Dict[str, List[float]],
    baseline_key: str = "vit_only",
) -> pd.DataFrame:
    """
    Create comparison table with all conditions
    
    Args:
        results_dict: {condition_name: [returns]}
        baseline_key: Name of baseline condition
    
    Returns:
        DataFrame with statistics
    """
    rows = []
    
    baseline_results = results_dict[baseline_key]
    
    for condition, results in results_dict.items():
        # Bootstrap CI
        mean, ci_low, ci_high = bootstrap_confidence_interval(
            np.array(results),
            statistic="mean"
        )
        
        row = {
            'Condition': condition,
            'Mean': f"{mean:.2f}",
            '95% CI': f"[{ci_low:.2f}, {ci_high:.2f}]",
            'Std': f"{np.std(results):.2f}",
            'Median': f"{np.median(results):.2f}",
        }
        
        # Compare to baseline
        if condition != baseline_key:
            comparison = compare_conditions(baseline_results, results)
            row['vs Baseline'] = f"{comparison['mean_diff']:+.2f}"
            row["Cohen's d"] = f"{comparison['cohens_d']:.3f}"
            row['p-value'] = f"{comparison['p_value']:.4f}"
            row['Significant'] = '✓' if comparison['significant'] else '✗'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df
```

---

## 6. Results Analysis Pipeline

### 6.1 Visualization Scripts

**File:** `evaluation/plot_results.py`
```python
"""
Generate plots for results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300


def plot_learning_curves(
    results_dict: Dict[str, Dict],
    env_name: str,
    save_path: str = None,
):
    """
    Plot learning curves for all conditions
    
    Args:
        results_dict: {condition: {'timesteps': [...], 'returns': [...]}}
        env_name: Environment name
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for condition, data in results_dict.items():
        timesteps = data['timesteps']
        returns = data['returns']
        
        # Plot mean and std across seeds
        ax.plot(timesteps, np.mean(returns, axis=0), label=condition, linewidth=2)
        ax.fill_between(
            timesteps,
            np.mean(returns, axis=0) - np.std(returns, axis=0),
            np.mean(returns, axis=0) + np.std(returns, axis=0),
            alpha=0.2
        )
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title(f'Learning Curves - {env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_final_performance_bars(
    results_dict: Dict[str, List[float]],
    env_name: str,
    save_path: str = None,
):
    """Bar plot of final performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = list(results_dict.keys())
    means = [np.mean(results_dict[c]) for c in conditions]
    stds = [np.std(results_dict[c]) for c in conditions]
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    
    # Color best performer
    best_idx = np.argmax(means)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Final Episode Return')
    ax.set_title(f'Final Performance - {env_name}')
    ax.grid(True, axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def plot_sample_efficiency(
    results_dict: Dict[str, Dict],
    threshold: float,
    env_name: str,
    save_path: str = None,
):
    """Plot steps to reach performance threshold"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = []
    steps_to_threshold = []
    
    for condition, data in results_dict.items():
        # Find when threshold is reached
        timesteps = data['timesteps']
        returns = np.mean(data['returns'], axis=0)
        
        idx = np.where(returns >= threshold)[0]
        if len(idx) > 0:
            steps = timesteps[idx[0]]
        else:
            steps = timesteps[-1]  # Didn't reach
        
        conditions.append(condition)
        steps_to_threshold.append(steps)
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, steps_to_threshold, alpha=0.8)
    
    # Highlight fastest
    fastest_idx = np.argmin(steps_to_threshold)
    bars[fastest_idx].set_color('green')
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Steps to Threshold')
    ax.set_title(f'Sample Efficiency - {env_name} (Threshold: {threshold})')
    ax.axhline(y=np.min(steps_to_threshold), color='gray', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def plot_robustness_heatmap(
    robustness_results: Dict[str, Dict[str, float]],
    save_path: str = None,
):
    """
    Heatmap showing performance under different robustness tests
    
    Args:
        robustness_results: {condition: {test_name: score}}
    """
    # Convert to DataFrame
    df = pd.DataFrame(robustness_results).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=df.mean().mean(),
        ax=ax,
        cbar_kws={'label': 'Performance'},
    )
    
    ax.set_title('Robustness Test Results')
    ax.set_xlabel('Test Condition')
    ax.set_ylabel('Model Condition')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def create_poster_figure(
    results_dict: Dict,
    env_name: str,
    save_path: str = None,
):
    """
    Create publication-quality figure for poster
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Learning curves
    ax1 = fig.add_subplot(gs[0, :2])
    # ... (plot learning curves)
    
    # 2. Final performance
    ax2 = fig.add_subplot(gs[0, 2])
    # ... (plot final performance bars)
    
    # 3. Sample efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    # ... (plot sample efficiency)
    
    # 4. Statistical comparison table
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('tight')
    ax4.axis('off')
    # ... (add comparison table)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
```

---

## 7. Timeline & Checklist

### Week 1: Infrastructure (Jan 27 - Feb 2)
- [ ] Day 1: Environment setup, dependencies installation
- [ ] Day 2: Implement environment wrappers and test
- [ ] Day 3: Implement ViT backbone and baseline policy
- [ ] Day 4: Implement PPO training loop (vision-only)
- [ ] Day 5: Run first baseline experiments on Pong
- [ ] Day 6: Verify training works, fix bugs
- [ ] Day 7: Buffer day / documentation

### Week 2: Multimodal Components (Feb 3 - Feb 9)
- [ ] Day 8: Implement text history buffer
- [ ] Day 9: Implement text encoder module
- [ ] Day 10: Implement fusion mechanisms (all 3 types)
- [ ] Day 11: Implement multimodal policy
- [ ] Day 12: Test multimodal forward pass
- [ ] Day 13: Integrate with PPO training
- [ ] Day 14: Run first multimodal experiment

### Week 3: Full Benchmarking (Feb 10 - Feb 16)
- [ ] Day 15-16: Run all conditions on Pong (5 seeds each)
- [ ] Day 17-18: Run all conditions on Breakout
- [ ] Day 19-20: Run robustness tests
- [ ] Day 21: Buffer for re-runs if needed

### Week 4: Analysis & Visualization (Feb 17 - Feb 23)
- [ ] Day 22: Collect all results, organize data
- [ ] Day 23: Statistical analysis
- [ ] Day 24: Generate all plots
- [ ] Day 25: Create comparison tables
- [ ] Day 26: Create poster figures
- [ ] Day 27: Write results summary
- [ ] Day 28: Final review and polish

---

## 8. Quick Start Commands

### 8.1 Installation
```bash
# Clone/setup project
git clone <your-repo>
cd vit-text-rl

# Create environment
python -m venv vit_rl_env
source vit_rl_env/bin/activate  # Windows: vit_rl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Run Single Experiment
```bash
# Train ViT-only baseline on Pong
python algos/train_ppo.py \
    --env=PongNoFrameskip-v4 \
    --condition=vit_only \
    --seed=42 \
    --output-dir=results/pong_vit_seed42

# Train multimodal on Pong
python algos/train_ppo.py \
    --env=PongNoFrameskip-v4 \
    --condition=vit_text_detailed \
    --seed=42 \
    --output-dir=results/pong_vit_text_seed42
```

### 8.3 Run Full Benchmark
```bash
# Run all conditions on Pong with 5 seeds
python scripts/run_full_benchmark.py \
    --envs pong \
    --config configs/experiments.yaml \
    --output-dir results/full_benchmark

# Run specific conditions only
python scripts/run_full_benchmark.py \
    --envs pong breakout \
    --conditions vit_only vit_text_detailed vit_text_crossattn \
    --config configs/experiments.yaml
```

### 8.4 Evaluate and Plot
```bash
# Evaluate all checkpoints
python evaluation/eval.py \
    --checkpoint-dir results/full_benchmark \
    --num-episodes 20 \
    --output-dir results/evaluation

# Generate plots
python evaluation/plot_results.py \
    --results-dir results/evaluation \
    --output-dir results/figures

# Statistical analysis
python evaluation/statistical_analysis.py \
    --results-dir results/evaluation \
    --baseline vit_only \
    --output results/stats.csv
```

---

## 9. Expected Outcomes & Benchmarks

### 9.1 Success Criteria

**Primary Metrics:**
1. **Sample Efficiency**: Multimodal should reach target performance in ≤ 80% steps vs baseline
2. **Final Performance**: Multimodal should achieve ≥ 10% higher final return
3. **Robustness**: Multimodal should show ≥ 15% better performance under partial observability
4. **Statistical Significance**: p < 0.05 across 5 seeds

**Secondary Metrics:**
1. **Fusion Comparison**: Cross-attention vs concat vs gated
2. **Text Style Impact**: Detailed vs compact vs narrative
3. **History Length**: Optimal N for temporal window

### 9.2 Baseline Performance Targets

```python
EXPECTED_PERFORMANCE = {
    "PongNoFrameskip-v4": {
        "vit_only": {"mean": 18.0, "std": 2.0, "steps_to_10": 500_000},
        "vit_text": {"mean": 20.0, "std": 1.5, "steps_to_10": 400_000},
    },
    "BreakoutNoFrameskip-v4": {
        "vit_only": {"mean": 350.0, "std": 50.0, "steps_to_100": 2_000_000},
        "vit_text": {"mean": 400.0, "std": 40.0, "steps_to_100": 1_500_000},
    },
}
```

---

## 10. Troubleshooting & Optimization

### 10.1 Common Issues

**Issue 1: Training Instability**
- Solution: Increase `freeze_vit_layers` to 10-11
- Solution: Reduce learning rate to 1e-4
- Solution: Add gradient clipping (`max_grad_norm=0.5`)

**Issue 2: Slow Training**
- Solution: Use mixed precision training (torch.amp)
- Solution: Increase batch size if GPU memory allows
- Solution: Use smaller ViT variant (vit_small_patch16_224)

**Issue 3: Text Encoder OOM**
- Solution: Use TinyBERT or DistilBERT instead of BERT-base
- Solution: Freeze text encoder completely
- Solution: Cache text encodings for repeated descriptions

**Issue 4: Poor Multimodal Performance**
- Solution: Try different fusion mechanisms
- Solution: Tune history length (try 3, 5, 10)
- Solution: Verify text descriptions are informative

### 10.2 Performance Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, value = policy(obs_vision, obs_text)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient checkpointing for ViT
model.encoder.vit.set_grad_checkpointing(enable=True)

# Compile model (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")
```

---

## 11. Final Deliverables Checklist

### For Poster
- [ ] Learning curve plots (all conditions, all envs)
- [ ] Final performance comparison table
- [ ] Sample efficiency bar chart
- [ ] Robustness heatmap
- [ ] Statistical significance markers
- [ ] Architecture diagram (ViT + Text + Fusion)
- [ ] Ablation study results
- [ ] Key insights summary (3-5 bullet points)

### For Repository
- [ ] Complete codebase with README
- [ ] Requirements.txt with exact versions
- [ ] Pre-trained model checkpoints
- [ ] Training logs and curves
- [ ] Evaluation results (CSV/JSON)
- [ ] Generated plots (high-res PNG/PDF)
- [ ] Configuration files for all experiments
- [ ] Statistical analysis results

### Documentation
- [ ] Implementation guide (this document)
- [ ] API documentation
- [ ] Training tutorial
- [ ] Reproducibility instructions
- [ ] Known issues and limitations

---

## Appendix A: Full File Structure

```
vit-text-rl/
├── README.md
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── experiments.yaml
│   ├── ppo_default.yaml
│   └── model_variants.yaml
│
├── envs/
│   ├── __init__.py
│   ├── atari_env.py
│   └── wrappers.py
│
├── models/
│   ├── __init__.py
│   ├── vit_policy.py
│   ├── text_encoder.py
│   ├── fusion.py
│   ├── vit_text_policy.py
│   └── cnn_baseline.py
│
├── algos/
│   ├── __init__.py
│   ├── ppo_vit.py
│   ├── ppo_vit_text.py
│   └── train_ppo.py
│
├── utils/
│   ├── __init__.py
│   ├── text_history.py
│   ├── logging_utils.py
│   └── checkpointing.py
│
├── evaluation/
│   ├── __init__.py
│   ├── eval.py
│   ├── metrics.py
│   ├── plot_results.py
│   └── statistical_analysis.py
│
├── scripts/
│   ├── run_full_benchmark.py
│   ├── train_single.sh
│   └── eval_all.sh
│
├── tests/
│   ├── test_envs.py
│   ├── test_models.py
│   └── test_training.py
│
└── results/
    ├── logs/
    ├── checkpoints/
    ├── evaluation/
    └── figures/
```

---

## Appendix B: Hardware Requirements

### Minimum Configuration
- GPU: NVIDIA GTX 1080 Ti (11GB VRAM)
- RAM: 16GB
- Storage: 50GB
- Training time: ~48 hours per condition (Pong, 10M steps)

### Recommended Configuration
- GPU: NVIDIA RTX 3090 / A100 (24GB+ VRAM)
- RAM: 32GB
- Storage: 100GB SSD
- Training time: ~12-18 hours per condition

### Cloud Options
- AWS: p3.2xlarge (V100, ~$3/hour)
- Google Cloud: n1-highmem-8 + T4 GPU (~$1.5/hour)
- Vast.ai: RTX 3090 instances (~$0.30-0.60/hour)

---

**Document Version:** 1.0  
**Last Updated:** January 26, 2026  
**Status:** Ready for Implementation  

*This pipeline provides everything needed to execute, benchmark, and analyze your multimodal RL project. Follow the checklist systematically, and you'll have comprehensive results for your poster.*
