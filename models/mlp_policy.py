"""
Simple MLP policy for non-visual environments like CartPole
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class MLPPolicy(nn.Module):
    """Simple MLP policy for vector observations"""
    
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Policy head
        self.policy_net = nn.Linear(hidden_dim, num_actions)
        
        # Value head
        self.value_net = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(obs)
        logits = self.policy_net(features)
        value = self.value_net(features)
        return logits, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(obs)
        return self.value_net(features)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
