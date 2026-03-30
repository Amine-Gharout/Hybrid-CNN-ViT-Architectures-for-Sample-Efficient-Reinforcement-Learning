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
        fusion_type: str = "concat",
        pretrained: bool = True,
        freeze_vit_layers: int = 8,
        freeze_text: bool = True,
    ):
        super().__init__()
        
        self.vision_encoder = ViTBackbone(
            model_name=vit_model,
            pretrained=pretrained,
            freeze_layers=freeze_vit_layers,
            embedding_dim=embedding_dim,
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_model,
            embedding_dim=embedding_dim,
            freeze=freeze_text,
        )
        
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
    
    def encode(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
    ) -> torch.Tensor:
        vision_features = self.vision_encoder(obs_vision)
        text_features = self.text_encoder(obs_text)
        fused_features = self.fusion(vision_features, text_features)
        return fused_features
    
    def forward(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(obs_vision, obs_text)
        logits = self.policy_net(features)
        value = self.value_net(features)
        return logits, value
    
    def get_value(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
    ) -> torch.Tensor:
        features = self.encode(obs_vision, obs_text)
        return self.value_net(features)
    
    def get_action_and_value(
        self,
        obs_vision: torch.Tensor,
        obs_text: List[str],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs_vision, obs_text)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
