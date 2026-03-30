"""
CNN Policy - Classical RL baseline (no ViT)
Standard convolutional network used in DQN/PPO for visual environments
"""
import torch
import torch.nn as nn
import numpy as np


class CNNPolicy(nn.Module):
    """
    Classic CNN policy similar to Nature DQN architecture
    This is the standard approach before Vision Transformers
    """
    
    def __init__(
        self,
        num_actions: int,
        input_channels: int = 3,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Classic CNN architecture (similar to Nature DQN)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 224, 224)
            cnn_out = self.cnn(dummy)
            cnn_out_size = cnn_out.shape[1]
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(cnn_out_size, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, num_actions)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CNN Policy created with {total_params:,} parameters")
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Special initialization for actor/critic heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def get_features(self, x):
        """Extract features from observations"""
        cnn_features = self.cnn(x)
        features = self.feature_layer(cnn_features)
        return features
    
    def get_value(self, x):
        """Get value estimate"""
        features = self.get_features(x)
        return self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value"""
        features = self.get_features(x)
        
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(features),
        )


class SmallCNNPolicy(nn.Module):
    """
    Smaller CNN for MinAtar (10x10 input resized to 224x224)
    More appropriate for simpler environments
    """
    
    def __init__(
        self,
        num_actions: int,
        input_channels: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Smaller CNN for MinAtar
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 224, 224)
            cnn_out = self.cnn(dummy)
            cnn_out_size = cnn_out.shape[1]
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(cnn_out_size, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, num_actions)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Small CNN Policy created with {total_params:,} parameters")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def get_features(self, x):
        cnn_features = self.cnn(x)
        features = self.feature_layer(cnn_features)
        return features
    
    def get_value(self, x):
        features = self.get_features(x)
        return self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        features = self.get_features(x)
        
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(features),
        )
