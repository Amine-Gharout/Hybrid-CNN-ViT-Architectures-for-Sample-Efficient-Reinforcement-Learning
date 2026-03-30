"""
C51 (Categorical DQN) with Hybrid CNN-ViT backbone  (v2 – optimised fusion).
Distributional RL: learns Z(s,a) as categorical over N_ATOMS bins.
Uses the same lightweight gated fusion as the DQN Hybrid v2.
"""
import torch
import torch.nn as nn
import numpy as np
import timm

N_ATOMS = 51
V_MIN = -10.0
V_MAX = 10.0


class C51GatedFusion(nn.Module):
    """Same lightweight gated fusion as DQN hybrid v2."""

    def __init__(self, cnn_dim: int = 256, vit_dim: int = 192,
                 output_dim: int = 256):
        super().__init__()
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, cnn_dim),
            nn.ReLU(),
        )
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.vit_norm = nn.LayerNorm(cnn_dim)
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim * 2, cnn_dim // 2),
            nn.ReLU(),
            nn.Linear(cnn_dim // 2, 1),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(cnn_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )
        with torch.no_grad():
            self.gate[-1].bias.fill_(0.85)

    def forward(self, cnn_feat, vit_feat):
        vit_proj = self.vit_proj(vit_feat)
        cnn_n = self.cnn_norm(cnn_feat)
        vit_n = self.vit_norm(vit_proj)
        combined = torch.cat([cnn_n, vit_n], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * cnn_n + (1.0 - alpha) * vit_n
        return self.output_proj(fused)


class C51HybridCNNViT(nn.Module):
    """
    Categorical DQN (C51) with Hybrid CNN-ViT backbone.

    Architecture
    ------------
    CNN Branch   : Nature CNN → FC → [B, embedding_dim]
    ViT Branch   : ViT-Tiny (frozen) → [B, vit_dim], grad-scaled ×0.1
    Fusion       : Lightweight gated fusion (CNN-biased) → [B, embedding_dim]
    Dist Heads   : Dueling distributional (value + advantage atoms)
    Output       : per-action atom probabilities [B, A, N_ATOMS]
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
        vit_grad_scale: float = 0.1,
        n_atoms: int = N_ATOMS,
        v_min: float = V_MIN,
        v_max: float = V_MAX,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.img_size = img_size
        self.vit_grad_scale = vit_grad_scale
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer(
            'support', torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # ---- CNN encoder ----
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

        # ---- ViT encoder ----
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

        # ---- Fusion ----
        self.fusion = C51GatedFusion(
            cnn_dim=embedding_dim,
            vit_dim=vit_embed_dim,
            output_dim=embedding_dim,
        )

        # ---- Distributional heads ----
        if dueling:
            self.advantage = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions * n_atoms),
            )
            self.value = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_atoms),
            )
        else:
            self.q_dist = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions * n_atoms),
            )

        self._init_weights()
        self._log_params()

    def _init_weights(self):
        for layer in self.cnn_encoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.cnn_fc.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        heads = [self.advantage, self.value] if self.dueling else [self.q_dist]
        for mod in heads:
            for layer in mod.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tag = "Dueling-C51-Hybrid" if self.dueling else "C51-Hybrid"
        print(f"[{tag}] {total:,} params ({trainable:,} trainable)")

    def get_features(self, x):
        cnn_feat = self.cnn_fc(self.cnn_encoder(x))
        vit_feat = self.vit(x)
        if self.training and self.vit_grad_scale < 1.0:
            vit_feat = vit_feat * self.vit_grad_scale + vit_feat.detach() * (1.0 - self.vit_grad_scale)
        return self.fusion(cnn_feat, vit_feat)

    def dist(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 5:
            obs = obs[:, -1]
        sz = self.img_size
        if obs.shape[-2:] != (sz, sz):
            obs = nn.functional.interpolate(obs, size=(sz, sz), mode="bilinear")

        features = self.get_features(obs)

        if self.dueling:
            adv = self.advantage(features).view(-1, self.num_actions, self.n_atoms)
            val = self.value(features).view(-1, 1, self.n_atoms)
            logits = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            logits = self.q_dist(features).view(-1, self.num_actions, self.n_atoms)

        return torch.softmax(logits, dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        probs = self.dist(obs)
        q = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q
