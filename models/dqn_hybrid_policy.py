"""
DQN Q-Network using Hybrid CNN-ViT backbone  (v2 – optimised fusion).
Combines CNN's local feature extraction with ViT's global reasoning
in a value-based (DQN) framework with optional Dueling architecture.
Optimised for 8 GB VRAM – uses 96×96 input by default.

Key changes vs v1:
  • Replaced heavy cross-attention fusion with lightweight gated fusion
  • Gate initialised to favour CNN (≈0.7) so noisy ViT doesn't hurt early
  • LayerNorm on each branch before fusion for gradient stability
  • ViT gradient scaling (×0.1) so pretrained features change slowly
  • Fewer fusion parameters → trains faster in low-data regime
"""
import torch
import torch.nn as nn
import numpy as np
import timm


class DQNGatedFusion(nn.Module):
    """Lightweight gated fusion: CNN-biased, ViT-enhanced.

    Instead of cross-attention (overkill on single vectors), we:
      1. Project both branches to same dim and normalise
      2. Concatenate → small MLP → sigmoid gate
      3. Gate is initialised CNN-biased (~0.7 CNN, ~0.3 ViT)
      4. Output projection with residual from CNN
    """

    def __init__(self, cnn_dim: int = 256, vit_dim: int = 192,
                 output_dim: int = 256):
        super().__init__()
        self.cnn_dim = cnn_dim

        # Align ViT features to CNN dimension
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, cnn_dim),
            nn.ReLU(),
        )

        # Normalise each branch independently for stable gating
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.vit_norm = nn.LayerNorm(cnn_dim)

        # Lightweight gate: concat → single hidden → 1 sigmoid
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim * 2, cnn_dim // 2),
            nn.ReLU(),
            nn.Linear(cnn_dim // 2, 1),
            # no Sigmoid here – we apply it in forward() after bias init
        )

        # Output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(cnn_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

        self._init_gate_bias()

    def _init_gate_bias(self):
        """Initialise gate to output ~0.7 (CNN-biased).
        sigmoid(0.85) ≈ 0.7, so bias the final linear layer."""
        with torch.no_grad():
            self.gate[-1].bias.fill_(0.85)  # sigmoid(0.85) ≈ 0.70

    def forward(self, cnn_feat: torch.Tensor,
                vit_feat: torch.Tensor) -> torch.Tensor:
        vit_proj = self.vit_proj(vit_feat)         # [B, cnn_dim]

        cnn_n = self.cnn_norm(cnn_feat)            # [B, cnn_dim]
        vit_n = self.vit_norm(vit_proj)            # [B, cnn_dim]

        combined = torch.cat([cnn_n, vit_n], dim=-1)  # [B, 2*cnn_dim]
        alpha = torch.sigmoid(self.gate(combined))     # [B, 1]  ∈ (0,1)

        fused = alpha * cnn_n + (1.0 - alpha) * vit_n  # [B, cnn_dim]
        return self.output_proj(fused)


class DQNHybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT Deep Q-Network  (v2 – optimised).

    Architecture
    ------------
    CNN Branch  : 3-layer conv → FC → [B, embedding_dim]
    ViT Branch  : ViT-Tiny (frozen early layers) → [B, vit_dim]
                  ViT gradients scaled ×0.1 for stability
    Fusion      : Lightweight gated fusion (CNN-biased init) → [B, embedding_dim]
    Q-Head      : Dueling (Value + Advantage) or standard MLP
    Output      : Q-values [B, num_actions]
    """

    def __init__(
        self,
        num_actions: int,
        cnn_channels=(32, 64, 64),
        vit_model: str = "vit_tiny_patch16_224",
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        pretrained: bool = True,
        freeze_vit_layers: int = 8,
        dueling: bool = True,
        img_size: int = 96,
        vit_grad_scale: float = 0.1,   # scale ViT gradients
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.img_size = img_size
        self.vit_grad_scale = vit_grad_scale

        # ==================================================
        # CNN encoder (IMPALA-style, fast local features)
        # ==================================================
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, cnn_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            cnn_out_size = self.cnn_encoder(dummy).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_size, embedding_dim),
            nn.ReLU(),
        )

        # ==================================================
        # ViT encoder (global context)
        # ==================================================
        self.vit = timm.create_model(
            vit_model, pretrained=pretrained, num_classes=0,
            img_size=img_size,
        )
        vit_embed_dim = self.vit.embed_dim

        if freeze_vit_layers > 0:
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
                self.vit.cls_token.requires_grad = False
            if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
                self.vit.pos_embed.requires_grad = False
            for i, block in enumerate(self.vit.blocks):
                if i < freeze_vit_layers:
                    for param in block.parameters():
                        param.requires_grad = False

        # ==================================================
        # Lightweight Gated Fusion (v2)
        # ==================================================
        self.fusion = DQNGatedFusion(
            cnn_dim=embedding_dim,
            vit_dim=vit_embed_dim,
            output_dim=embedding_dim,
        )

        # ==================================================
        # Q-value heads
        # ==================================================
        if dueling:
            self.advantage = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )
            self.value = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )

        self._init_weights()
        self._log_params()

    # ------------------------------------------------------------------ utils
    def _init_weights(self):
        """Xavier init for CNN, fusion, and Q-heads."""
        init_modules = [self.cnn_encoder, self.cnn_fc]
        if self.dueling:
            init_modules += [self.advantage, self.value]
        else:
            init_modules += [self.q_head]

        for mod in init_modules:
            for layer in mod.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tag = "Dueling-DQN-Hybrid" if self.dueling else "DQN-Hybrid"
        print(f"[{tag}] {total:,} params ({trainable:,} trainable)")

    # ---------------------------------------------------------------- forward
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fused CNN-ViT features with ViT gradient scaling."""
        cnn_feat = self.cnn_fc(self.cnn_encoder(x))

        vit_feat = self.vit(x)
        # Scale ViT gradients so pretrained features change slowly
        if self.training and self.vit_grad_scale < 1.0:
            vit_feat = vit_feat * self.vit_grad_scale + vit_feat.detach() * (1.0 - self.vit_grad_scale)

        return self.fusion(cnn_feat, vit_feat)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor [B, 3, H, W]

        Returns
        -------
        q_values : Tensor [B, num_actions]
        """
        if obs.dim() == 5:
            obs = obs[:, -1]

        sz = self.img_size
        if obs.shape[-2:] != (sz, sz):
            obs = nn.functional.interpolate(obs, size=(sz, sz), mode="bilinear")

        features = self.get_features(obs)  # [B, embedding_dim]

        if self.dueling:
            advantage = self.advantage(features)
            value = self.value(features)
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q = self.q_head(features)

        return q
