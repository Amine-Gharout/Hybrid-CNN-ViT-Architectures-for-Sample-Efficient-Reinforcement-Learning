"""
C51 (Categorical DQN) with pure CNN backbone.
Instead of scalar Q(s,a), learns the return distribution Z(s,a)
as a categorical distribution over N_ATOMS bins.
"""
import torch
import torch.nn as nn
import numpy as np

N_ATOMS = 51
V_MIN = -10.0
V_MAX = 10.0


class C51CNN(nn.Module):
    """
    Categorical DQN (C51) with Nature-DQN CNN backbone.

    Architecture
    ------------
    Input (3×img_size×img_size)
      → Nature CNN → FC → embedding_dim → ReLU
      → Dueling distributional heads:
          Value:     [B, 1, N_ATOMS]
          Advantage: [B, A, N_ATOMS]
      → Softmax → per-action atom probabilities [B, A, N_ATOMS]
    """

    def __init__(
        self,
        num_actions: int,
        cnn_channels=(32, 64, 64),
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        dueling: bool = True,
        img_size: int = 96,
        n_atoms: int = N_ATOMS,
        v_min: float = V_MIN,
        v_max: float = V_MAX,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.img_size = img_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Atom support
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

        self.feature_layer = nn.Sequential(
            nn.Linear(cnn_out_size, embedding_dim),
            nn.ReLU(),
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
        for layer in self.feature_layer.modules():
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
        tag = "Dueling-C51-CNN" if self.dueling else "C51-CNN"
        print(f"[{tag}] {total:,} params ({trainable:,} trainable)")

    def dist(self, obs: torch.Tensor) -> torch.Tensor:
        """Return per-action atom probabilities [B, A, N_ATOMS]."""
        if obs.dim() == 5:
            obs = obs[:, -1]
        sz = self.img_size
        if obs.shape[-2:] != (sz, sz):
            obs = nn.functional.interpolate(obs, size=(sz, sz), mode="bilinear")

        features = self.feature_layer(self.cnn_encoder(obs))

        if self.dueling:
            adv = self.advantage(features).view(-1, self.num_actions, self.n_atoms)
            val = self.value(features).view(-1, 1, self.n_atoms)
            logits = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            logits = self.q_dist(features).view(-1, self.num_actions, self.n_atoms)

        return torch.softmax(logits, dim=-1)  # [B, A, N_ATOMS]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values [B, A] for action selection."""
        probs = self.dist(obs)  # [B, A, N_ATOMS]
        q = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, A]
        return q
