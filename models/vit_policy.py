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
        freeze_layers: int = 8,
        embedding_dim: int = 768,
    ):
        super().__init__()
        
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        self.embedding_dim = embedding_dim
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            vit_output_dim = self.vit(dummy_input).shape[-1]
        
        if vit_output_dim != embedding_dim:
            self.projection = nn.Linear(vit_output_dim, embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def _freeze_layers(self, num_layers: int):
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        
        self.vit.cls_token.requires_grad = False
        self.vit.pos_embed.requires_grad = False
        
        if hasattr(self.vit, 'blocks'):
            for i, block in enumerate(self.vit.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 1D observations (like CartPole)
        if x.dim() == 2:  # [B, features]
            # Skip ViT, just use projection
            return self.projection(x) if isinstance(self.projection, nn.Linear) else x
        
        # Handle frame stacking
        if x.dim() == 5:
            x = x[:, -1]
        
        if x.shape[-2:] != (224, 224):
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        
        features = self.vit(x)
        embedding = self.projection(features)
        
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
        super().__init__()
        
        self.encoder = ViTBackbone(
            model_name=vit_model,
            pretrained=pretrained,
            freeze_layers=freeze_vit_layers,
            embedding_dim=embedding_dim,
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        logits = self.policy_net(features)
        value = self.value_net(features)
        return logits, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
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
