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
        super().__init__()
        
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
        combined = torch.cat([vision_features, text_features], dim=-1)
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
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
        )
        
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        vision_seq = vision_proj.unsqueeze(1)
        text_seq = text_proj.unsqueeze(1)
        
        attn_output, _ = self.cross_attention(
            query=vision_seq,
            key=text_seq,
            value=text_seq,
        )
        
        fused = self.norm1(vision_proj + self.dropout(attn_output.squeeze(1)))
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
        super().__init__()
        
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(vision_dim + text_dim, output_dim),
            nn.Sigmoid(),
        )
        
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        gate_input = torch.cat([vision_features, text_features], dim=-1)
        gate_weight = self.gate(gate_input)
        
        fused = gate_weight * vision_proj + (1 - gate_weight) * text_proj
        output = self.output_proj(fused)
        
        return output
