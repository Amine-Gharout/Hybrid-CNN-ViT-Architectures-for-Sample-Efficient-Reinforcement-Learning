"""
C51 (Categorical DQN) with ViT backbone.
Distributional RL: learns Z(s,a) as categorical over N_ATOMS bins.
"""
import torch
import torch.nn as nn
import timm

N_ATOMS = 51
V_MIN = -10.0
V_MAX = 10.0


class C51ViT(nn.Module):
    """
    Categorical DQN (C51) with Vision Transformer backbone.

    Architecture
    ------------
    Input (3×img_size×img_size)
      → ViT backbone (frozen early layers) → projection → embedding_dim
      → Dueling distributional heads
      → Softmax → per-action atom probabilities [B, A, N_ATOMS]
    """

    def __init__(
        self,
        num_actions: int,
        vit_model: str = "vit_tiny_patch16_224",
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        pretrained: bool = True,
        freeze_vit_layers: int = 10,
        dueling: bool = True,
        img_size: int = 96,
        n_atoms: int = N_ATOMS,
        v_min: float = V_MIN,
        v_max: float = V_MAX,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.img_size = img_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer(
            'support', torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # ---- ViT encoder ----
        self.vit = timm.create_model(
            vit_model, pretrained=pretrained, num_classes=0,
            img_size=img_size,
        )
        vit_out_dim = self.vit.embed_dim

        if freeze_vit_layers > 0:
            self._freeze_layers(freeze_vit_layers)

        self.projection = (
            nn.Linear(vit_out_dim, embedding_dim)
            if vit_out_dim != embedding_dim else nn.Identity()
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

    def _freeze_layers(self, num_layers):
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
            self.vit.cls_token.requires_grad = False
        if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
            self.vit.pos_embed.requires_grad = False
        for i, block in enumerate(self.vit.blocks):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def _init_weights(self):
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
        tag = "Dueling-C51-ViT" if self.dueling else "C51-ViT"
        print(f"[{tag}] {total:,} params ({trainable:,} trainable)")

    def dist(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 5:
            obs = obs[:, -1]
        sz = self.img_size
        if obs.shape[-2:] != (sz, sz):
            obs = nn.functional.interpolate(obs, size=(sz, sz), mode="bilinear")

        features = self.projection(self.vit(obs))

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
