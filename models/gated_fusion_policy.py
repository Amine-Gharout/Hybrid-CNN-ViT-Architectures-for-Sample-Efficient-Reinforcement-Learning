"""
Novel: Gated Adaptive Vision-Language Fusion Policy
Research contribution: Learns dynamic gating to fuse visual and textual information
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers import AutoTokenizer, AutoModel


class GatedFusionBlock(nn.Module):
    """Learns when to trust vision vs. text using dynamic gating"""
    
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        
        # Gate network - learns importance weights
        self.gate_net = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [vision_weight, text_weight]
            nn.Softmax(dim=-1)
        )
        
        # Feature projection to common space
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
    def forward(self, vision_feat, text_feat):
        """
        Dynamic fusion with learned gating
        Returns: fused_features, gate_weights (for interpretability)
        """
        # Concatenate for gate
        combined = torch.cat([vision_feat, text_feat], dim=-1)
        gate_weights = self.gate_net(combined)  # [B, 2]
        
        # Project to common space
        v_proj = self.vision_proj(vision_feat)
        t_proj = self.text_proj(text_feat)
        
        # Weighted fusion
        vision_weight = gate_weights[:, 0:1]  # [B, 1]
        text_weight = gate_weights[:, 1:2]    # [B, 1]
        
        fused = vision_weight * v_proj + text_weight * t_proj
        
        return fused, gate_weights


class LightweightCNN(nn.Module):
    """Fast CNN encoder for visual features"""
    
    def __init__(self, num_channels: int = 3):
        super().__init__()
        
        self.conv_net = nn.Sequential(
            # First conv block
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, 84, 84)
            self.feature_dim = self.conv_net(dummy).shape[1]
    
    def forward(self, x):
        return self.conv_net(x)


class TextEncoder(nn.Module):
    """Lightweight text encoder using pretrained embeddings"""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        
        # Use lightweight sentence transformer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Freeze text encoder for speed
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Project to desired dimension
        self.projection = nn.Linear(384, embedding_dim)  # MiniLM outputs 384
        self.embedding_dim = embedding_dim
        
    def forward(self, texts: List[str]):
        """Encode text instructions"""
        device = next(self.projection.parameters()).device
        
        # Tokenize
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=128
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.text_model(**encoded)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Project
        return self.projection(embeddings)


class GatedVisionLanguagePolicy(nn.Module):
    """
    NOVEL ARCHITECTURE: Gated Adaptive Vision-Language Fusion
    
    Key Innovation: Learns dynamic gating to determine when to trust:
    - Visual observations (for reactive behavior)
    - Text instructions (for strategic guidance)
    
    This allows the policy to adaptively balance perception and instruction.
    """
    
    def __init__(
        self,
        num_actions: int,
        vision_channels: int = 3,
        hidden_dim: int = 512,
        text_dim: int = 256,
    ):
        super().__init__()
        
        print("=" * 60)
        print("GATED VISION-LANGUAGE FUSION POLICY")
        print("=" * 60)
        
        # Vision encoder (fast CNN)
        self.vision_encoder = LightweightCNN(vision_channels)
        vision_dim = self.vision_encoder.feature_dim
        
        # Text encoder (frozen pretrained)
        self.text_encoder = TextEncoder(text_dim)
        
        # NOVEL: Gated fusion mechanism
        self.gated_fusion = GatedFusionBlock(vision_dim, text_dim, hidden_dim)
        
        # Policy head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Value head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Track gating statistics for analysis
        self.register_buffer('vision_gate_avg', torch.tensor(0.0))
        self.register_buffer('text_gate_avg', torch.tensor(0.0))
        self.register_buffer('gate_count', torch.tensor(0))
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params:,}")
        print(f"Vision encoder dim: {vision_dim}")
        print(f"Text encoder dim: {text_dim}")
        print(f"Fusion dim: {hidden_dim}")
        print("=" * 60)
    
    def forward(self, obs, text_instructions: List[str]):
        """Forward pass with gated fusion"""
        # Encode vision
        vision_feat = self.vision_encoder(obs)
        
        # Encode text (cached for same instruction)
        text_feat = self.text_encoder(text_instructions)
        
        # NOVEL: Gated fusion
        fused_feat, gate_weights = self.gated_fusion(vision_feat, text_feat)
        
        # Update gate statistics
        self.vision_gate_avg = 0.99 * self.vision_gate_avg + 0.01 * gate_weights[:, 0].mean()
        self.text_gate_avg = 0.99 * self.text_gate_avg + 0.01 * gate_weights[:, 1].mean()
        self.gate_count += 1
        
        # Policy and value
        logits = self.actor(fused_feat)
        value = self.critic(fused_feat)
        
        return logits, value, gate_weights
    
    def get_value(self, obs, text_instructions: List[str]):
        """Get value estimate"""
        _, value, _ = self.forward(obs, text_instructions)
        return value
    
    def get_action_and_value(
        self, 
        obs, 
        text_instructions: List[str],
        action: Optional[torch.Tensor] = None
    ):
        """Get action and value with gated fusion"""
        logits, value, gate_weights = self.forward(obs, text_instructions)
        
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_gate_statistics(self):
        """Get average gating weights for analysis"""
        return {
            'vision_weight': self.vision_gate_avg.item(),
            'text_weight': self.text_gate_avg.item(),
            'samples': self.gate_count.item()
        }


class FastHybridCNN(nn.Module):
    """Faster hybrid with multi-scale fusion - no ViT"""
    
    def __init__(self, num_actions: int):
        super().__init__()
        
        # Multi-scale CNN
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        
        # Calculate feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            x = torch.relu(self.conv1(dummy))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            feat_size = x.view(1, -1).shape[1]
        
        self.fc = nn.Linear(feat_size, 512)
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)
        
        print(f"Fast Hybrid CNN created with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, obs):
        x = torch.relu(self.conv1(obs))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.actor(x), self.critic(x)
    
    def get_value(self, obs):
        _, value = self.forward(obs)
        return value
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
