"""
Hybrid CNN-ViT Policy for Reinforcement Learning
Combines CNN's efficiency with ViT's global reasoning through adaptive fusion
"""
import torch
import torch.nn as nn
import numpy as np
import timm


class AdaptiveFusion(nn.Module):
    """Learned gating between CNN and ViT features"""
    
    def __init__(self, cnn_dim=512, vit_dim=192, output_dim=512):
        super().__init__()
        
        # Project ViT to match CNN dimension
        self.vit_proj = nn.Linear(vit_dim, cnn_dim)
        
        # Cross-attention: ViT attends to CNN
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Gating mechanism - learns when to use CNN vs ViT
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim * 2, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(cnn_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, cnn_feat, vit_feat):
        # Project ViT features to CNN dimension
        vit_proj = self.vit_proj(vit_feat)  # [B, cnn_dim]
        
        # Cross-attention: use ViT as query, CNN as key/value
        attn_out, _ = self.cross_attn(
            vit_proj.unsqueeze(1),  # query: [B, 1, D]
            cnn_feat.unsqueeze(1),  # key: [B, 1, D]
            cnn_feat.unsqueeze(1)   # value: [B, 1, D]
        )
        attn_out = attn_out.squeeze(1)  # [B, D]
        
        # Compute adaptive gate
        combined = torch.cat([cnn_feat, attn_out], dim=-1)
        gate = self.gate(combined)  # [B, 1] - how much to use CNN vs attention
        
        # Gated fusion
        fused = gate * cnn_feat + (1 - gate) * attn_out
        
        return self.output_proj(fused)


class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT Actor-Critic Policy
    
    Architecture:
    - CNN Branch: Fast, translation-invariant local features
    - ViT Branch: Global context, long-range dependencies
    - Adaptive Fusion: Learned gating between both
    """
    
    def __init__(
        self,
        num_actions: int,
        cnn_channels=[32, 64, 64],
        vit_model="vit_tiny_patch16_224",
        embedding_dim=512,
        hidden_dim=256,
        pretrained=True,
        freeze_vit_layers=8,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        
        # ==========================================
        # CNN Encoder (IMPALA-style, fast & stable)
        # ==========================================
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, cnn_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            cnn_out = self.cnn_encoder(dummy)
            cnn_out_size = cnn_out.shape[1]
        
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_size, embedding_dim),
            nn.ReLU(),
        )
        
        # ==========================================
        # ViT Encoder (global context)
        # ==========================================
        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        vit_embed_dim = self.vit.embed_dim
        
        # Freeze early ViT layers for stability
        if freeze_vit_layers > 0:
            # Freeze patch embedding
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            
            # Freeze early transformer blocks
            for i, block in enumerate(self.vit.blocks):
                if i < freeze_vit_layers:
                    for param in block.parameters():
                        param.requires_grad = False
        
        # ==========================================
        # Adaptive Fusion Module
        # ==========================================
        self.fusion = AdaptiveFusion(
            cnn_dim=embedding_dim,
            vit_dim=vit_embed_dim,
            output_dim=embedding_dim
        )
        
        # ==========================================
        # Actor-Critic Heads
        # ==========================================
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Hybrid CNN-ViT created: {total:,} params ({trainable:,} trainable)")
    
    def _init_weights(self):
        """Orthogonal initialization for RL stability"""
        for module in [self.cnn_encoder, self.cnn_fc, self.actor, self.critic]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Small init for policy output (better exploration)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        
        # Normal init for value output
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)
    
    def get_features(self, x):
        """Extract fused CNN-ViT features"""
        # CNN branch (fast, local)
        cnn_feat = self.cnn_encoder(x)
        cnn_feat = self.cnn_fc(cnn_feat)
        
        # ViT branch (slow, global)
        vit_feat = self.vit(x)
        
        # Adaptive fusion
        fused = self.fusion(cnn_feat, vit_feat)
        
        return fused
    
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


class HybridCNNViTSmall(nn.Module):
    """
    Smaller Hybrid for faster training on MinAtar
    Uses smaller CNN and ViT-Tiny
    """
    
    def __init__(
        self,
        num_actions: int,
        embedding_dim=256,
        hidden_dim=128,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Small CNN
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            cnn_out = self.cnn_encoder(dummy)
            cnn_out_size = cnn_out.shape[1]
        
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_size, embedding_dim),
            nn.ReLU(),
        )
        
        # ViT-Tiny
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0,
        )
        vit_embed_dim = self.vit.embed_dim  # 192
        
        # Freeze most of ViT
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        for i, block in enumerate(self.vit.blocks):
            if i < 10:  # Freeze 10 of 12 blocks
                for param in block.parameters():
                    param.requires_grad = False
        
        # Simple fusion (concat + project)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + vit_embed_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim + vit_embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Heads
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Hybrid Small created: {total:,} params ({trainable:,} trainable)")
    
    def _init_weights(self):
        for module in [self.cnn_encoder, self.cnn_fc, self.fusion, self.actor, self.critic]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def get_features(self, x):
        # CNN features
        cnn_feat = self.cnn_fc(self.cnn_encoder(x))
        
        # ViT features
        vit_feat = self.vit(x)
        
        # Concat and compute gate
        combined = torch.cat([cnn_feat, vit_feat], dim=-1)
        gate = self.gate(combined)
        
        # Fused representation
        fused = self.fusion(combined)
        
        # Apply gate (weighted combination)
        # gate=1 means more CNN influence, gate=0 means more ViT
        output = gate * cnn_feat + (1 - gate) * fused
        
        return output
    
    def get_value(self, x):
        return self.critic(self.get_features(x))
    
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
