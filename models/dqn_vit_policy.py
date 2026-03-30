"""
DQN Q-Network using Vision Transformer backbone.
Supports standard DQN and Dueling DQN architectures.
Optimised for 8 GB VRAM – uses 96×96 input by default (36 patches).
"""
import torch
import torch.nn as nn
import timm
from typing import Optional


class DQNViT(nn.Module):
    """
    Deep Q-Network with a Vision Transformer encoder.

    Architecture
    ------------
    Input (3×img_size×img_size)  [default 96×96]
      → ViT backbone (patch embed + transformer blocks)
      → CLS token projection (→ embedding_dim)
      → Q-head  (standard or dueling)
      → Q-values [B, num_actions]
    """

    def __init__(
        self,
        num_actions: int,
        vit_model: str = "vit_base_patch16_224",
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_vit_layers: int = 8,
        dueling: bool = True,
        img_size: int = 96,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.img_size = img_size

        # ---- ViT encoder ----
        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,  # remove classifier
            img_size=img_size,
        )

        # Freeze early layers for stability
        if freeze_vit_layers > 0:
            self._freeze_layers(freeze_vit_layers)

        # Determine output dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            vit_out_dim = self.vit(dummy).shape[-1]

        # Projection to fixed embedding size
        self.projection = (
            nn.Linear(vit_out_dim, embedding_dim)
            if vit_out_dim != embedding_dim
            else nn.Identity()
        )

        # ---- Q-value heads ----
        if dueling:
            # Advantage stream
            self.advantage = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )
            # Value stream
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
    def _freeze_layers(self, num_layers: int):
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
            self.vit.cls_token.requires_grad = False
        if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
            self.vit.pos_embed.requires_grad = False
        if hasattr(self.vit, 'blocks'):
            for i, block in enumerate(self.vit.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False

    def _init_weights(self):
        """Xavier init for Q-heads (standard for value-based methods)."""
        modules = (
            [self.advantage, self.value] if self.dueling else [self.q_head]
        )
        for mod in modules:
            for layer in mod.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tag = "Dueling-DQN-ViT" if self.dueling else "DQN-ViT"
        print(f"[{tag}] {total:,} params ({trainable:,} trainable)")

    # ---------------------------------------------------------------- forward
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor [B, 3, H, W]

        Returns
        -------
        q_values : Tensor [B, num_actions]
        """
        # Handle frame-stacked input [B, T, C, H, W]
        if obs.dim() == 5:
            obs = obs[:, -1]

        sz = self.img_size
        if obs.shape[-2:] != (sz, sz):
            obs = nn.functional.interpolate(obs, size=(sz, sz), mode="bilinear")

        features = self.projection(self.vit(obs))  # [B, embedding_dim]

        if self.dueling:
            advantage = self.advantage(features)          # [B, A]
            value = self.value(features)                  # [B, 1]
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q = self.q_head(features)

        return q
