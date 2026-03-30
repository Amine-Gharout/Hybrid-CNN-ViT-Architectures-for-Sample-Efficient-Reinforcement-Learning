"""
DQN Q-Network using a pure CNN backbone (Nature DQN style).
Serves as the classical baseline to compare with ViT and Hybrid.
Optimised for 8 GB VRAM – uses 96×96 input by default.
"""
import torch
import torch.nn as nn
import numpy as np


class DQNCNN(nn.Module):
    """
    Deep Q-Network with a classical CNN encoder (Nature DQN architecture).

    Architecture
    ------------
    Input (3×img_size×img_size)  [default 96×96]
      → Conv 8×8 s4 → 32 filters → ReLU
      → Conv 4×4 s2 → 64 filters → ReLU
      → Conv 3×3 s1 → 64 filters → ReLU
      → Flatten → FC → embedding_dim → ReLU
      → Q-head  (standard or dueling)
      → Q-values [B, num_actions]
    """

    def __init__(
        self,
        num_actions: int,
        cnn_channels=(32, 64, 64),
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        dueling: bool = True,
        img_size: int = 96,
        # Accept (and ignore) ViT-specific kwargs so the benchmark runner
        # can pass the same kwargs dict for all models without errors.
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.img_size = img_size

        # ---- CNN encoder (Nature DQN) ----
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, cnn_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            cnn_out_size = self.cnn_encoder(dummy).shape[1]

        # Feature projection
        self.feature_layer = nn.Sequential(
            nn.Linear(cnn_out_size, embedding_dim),
            nn.ReLU(),
        )

        # ---- Q-value heads ----
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
        """Orthogonal init for CNN, Xavier for Q-heads."""
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

        q_modules = (
            [self.advantage, self.value] if self.dueling else [self.q_head]
        )
        for mod in q_modules:
            for layer in mod.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tag = "Dueling-DQN-CNN" if self.dueling else "DQN-CNN"
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
        if obs.dim() == 5:
            obs = obs[:, -1]

        sz = self.img_size
        if obs.shape[-2:] != (sz, sz):
            obs = nn.functional.interpolate(obs, size=(sz, sz), mode="bilinear")

        features = self.feature_layer(self.cnn_encoder(obs))

        if self.dueling:
            advantage = self.advantage(features)
            value = self.value(features)
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q = self.q_head(features)

        return q
