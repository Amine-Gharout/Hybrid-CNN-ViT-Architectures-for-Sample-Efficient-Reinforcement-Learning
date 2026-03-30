"""
DQN Training Script for ViT and Hybrid CNN-ViT Q-Networks
Uses Double DQN with target network and experience replay.
"""
import argparse
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from envs.atari_env import make_parallel_envs
from models.dqn_vit_policy import DQNViT
from models.dqn_hybrid_policy import DQNHybridCNNViT
from evaluation.metrics import MetricsCollector


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN training.
    Stores observations as uint8 to save 4× RAM."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Pre-allocate storage (uint8 for obs → 4× less RAM)
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)

    def push(self, obs, action, reward, next_obs, done):
        """Add a single transition (obs should be uint8 0-255)."""
        self.obs[self.pos] = torch.as_tensor(obs, dtype=torch.uint8)
        self.next_obs[self.pos] = torch.as_tensor(next_obs, dtype=torch.uint8)
        self.actions[self.pos] = torch.as_tensor(action, dtype=torch.long)
        self.rewards[self.pos] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones[self.pos] = torch.as_tensor(done, dtype=torch.float32)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, obs, actions, rewards, next_obs, dones):
        """Add a batch of transitions from parallel envs."""
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

    def sample(self, batch_size: int):
        """Sample a random batch, converting uint8→float32 on the fly."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs].to(self.device, dtype=torch.float32).div_(255.0),
            self.actions[idxs].to(self.device),
            self.rewards[idxs].to(self.device),
            self.next_obs[idxs].to(self.device, dtype=torch.float32).div_(255.0),
            self.dones[idxs].to(self.device),
        )

    def __len__(self):
        return self.size


# =============================================================================
# Epsilon Schedule
# =============================================================================

class LinearEpsilonSchedule:
    """Linear annealing from eps_start to eps_end over schedule_steps."""

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.01,
                 schedule_steps: int = 100_000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_steps = schedule_steps

    def __call__(self, step: int) -> float:
        frac = min(1.0, step / self.schedule_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DQN Training with ViT / Hybrid-ViT")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--condition", type=str, default="dqn_vit",
                        choices=["dqn_vit", "dqn_hybrid"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/dqn_experiments.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Policy factory
# =============================================================================

def create_q_network(condition: str, config: Dict, num_actions: int, device: str):
    """Instantiate the appropriate Q-network."""
    model_cfg = config["model"]

    if condition == "dqn_vit":
        cond_cfg = config["conditions"]["dqn_vit"]
        net = DQNViT(
            num_actions=num_actions,
            vit_model=cond_cfg.get("vit_model", model_cfg["vit_model"]),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            pretrained=cond_cfg.get("pretrained", True),
            freeze_vit_layers=model_cfg["freeze_vit_layers"],
            dueling=cond_cfg.get("dueling", True),
        ).to(device)
    elif condition == "dqn_hybrid":
        cond_cfg = config["conditions"]["dqn_hybrid"]
        net = DQNHybridCNNViT(
            num_actions=num_actions,
            vit_model=cond_cfg.get("vit_model", model_cfg.get("hybrid_vit_model", "vit_tiny_patch16_224")),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            pretrained=cond_cfg.get("pretrained", True),
            freeze_vit_layers=model_cfg["freeze_vit_layers"],
            dueling=cond_cfg.get("dueling", True),
        ).to(device)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return net


# =============================================================================
# Main training loop
# =============================================================================

def train(args):
    # Load config
    config = load_config(args.config)
    train_cfg = config["training"]
    set_seed(args.seed)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional W&B
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   config={**config, "env": args.env, "seed": args.seed,
                           "condition": args.condition})

    # TensorBoard
    writer = SummaryWriter(str(output_dir / "tensorboard"))

    # ------- Environment -------
    envs = make_parallel_envs(
        env_name=args.env,
        num_envs=train_cfg["num_envs"],
        seed=args.seed,
        vit_mode=True,
    )
    num_actions = envs.action_space.n if hasattr(envs.action_space, 'n') else int(envs.action_space.nvec[0])
    obs_space = envs.single_observation_space if hasattr(envs, 'single_observation_space') else envs.observation_space
    obs_shape = (3, 224, 224)

    print(f"Environment : {args.env}")
    print(f"  Obs space : {obs_space}")
    print(f"  Actions   : {num_actions}")
    print(f"  Condition : {args.condition}")

    # ------- Networks -------
    online_net = create_q_network(args.condition, config, num_actions, args.device)
    target_net = create_q_network(args.condition, config, num_actions, args.device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    total_params = sum(p.numel() for p in online_net.parameters())
    train_params = sum(p.numel() for p in online_net.parameters() if p.requires_grad)
    print(f"  Params    : {total_params:,} total, {train_params:,} trainable")

    # ------- Optimiser -------
    optimizer = optim.Adam(
        online_net.parameters(),
        lr=train_cfg["learning_rate"],
        eps=train_cfg.get("adam_epsilon", 1e-5),
    )

    # ------- Replay buffer -------
    replay = ReplayBuffer(
        capacity=train_cfg["replay_buffer_size"],
        obs_shape=obs_shape,
        device=args.device,
    )

    # ------- Schedules -------
    eps_schedule = LinearEpsilonSchedule(
        eps_start=train_cfg["eps_start"],
        eps_end=train_cfg["eps_end"],
        schedule_steps=train_cfg["eps_schedule_steps"],
    )

    # ------- Metrics -------
    metrics_collector = MetricsCollector()
    recent_returns = deque(maxlen=100)
    num_envs = train_cfg["num_envs"]

    # ------- Init env -------
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(args.device)

    total_timesteps = train_cfg["total_timesteps"]
    learning_starts = train_cfg["learning_starts"]
    train_freq = train_cfg["train_frequency"]
    batch_size = train_cfg["batch_size"]
    gamma = train_cfg["gamma"]
    target_update_freq = train_cfg["target_update_frequency"]
    use_double_dqn = train_cfg.get("double_dqn", True)

    print(f"  Total steps   : {total_timesteps:,}")
    print(f"  Replay size   : {train_cfg['replay_buffer_size']:,}")
    print(f"  Learn starts  : {learning_starts:,}")
    print(f"  Double DQN    : {use_double_dqn}")
    print(f"  Device        : {args.device}")
    print()

    # ============================================================
    # Training loop
    # ============================================================
    num_updates = 0
    while global_step < total_timesteps:
        # Epsilon-greedy action selection
        epsilon = eps_schedule(global_step)
        with torch.no_grad():
            q_values = online_net(next_obs)  # [num_envs, num_actions]

        # Vectorised eps-greedy
        rand_mask = torch.rand(num_envs, device=args.device) < epsilon
        random_actions = torch.randint(0, num_actions, (num_envs,), device=args.device)
        greedy_actions = q_values.argmax(dim=1)
        actions = torch.where(rand_mask, random_actions, greedy_actions)

        # Step environments
        new_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        dones = np.logical_or(terminated, truncated)
        new_obs_t = torch.Tensor(new_obs).to(args.device)

        # Store transitions
        replay.push_batch(
            next_obs.cpu(), actions.cpu(),
            torch.tensor(rewards, dtype=torch.float32),
            new_obs_t.cpu(),
            torch.tensor(dones, dtype=torch.float32),
        )

        next_obs = new_obs_t
        global_step += num_envs

        # Log episode completions
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_ret = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    recent_returns.append(ep_ret)
                    writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)
                    if args.wandb_project:
                        import wandb
                        wandb.log({"episodic_return": ep_ret, "global_step": global_step})

        # ----- Training step -----
        if global_step >= learning_starts and global_step % train_freq == 0:
            s_obs, s_actions, s_rewards, s_next_obs, s_dones = replay.sample(batch_size)

            with torch.no_grad():
                if use_double_dqn:
                    # Double DQN: online net picks action, target evaluates
                    next_q_online = online_net(s_next_obs)
                    best_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(s_next_obs)
                    next_q = next_q_target.gather(1, best_actions).squeeze(1)
                else:
                    next_q = target_net(s_next_obs).max(dim=1).values

                td_target = s_rewards + gamma * next_q * (1.0 - s_dones)

            current_q = online_net(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.huber_loss(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(),
                                     train_cfg.get("max_grad_norm", 10.0))
            optimizer.step()
            num_updates += 1

            # Update target network
            if num_updates % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            # Logging
            if num_updates % train_cfg.get("log_interval", 100) == 0:
                sps = int(global_step / (time.time() - start_time))
                mean_ret = np.mean(recent_returns) if recent_returns else 0.0
                print(f"Step {global_step:>8,} | Updates {num_updates:>6,} | "
                      f"Loss {loss.item():.4f} | Eps {epsilon:.3f} | "
                      f"Mean100 {mean_ret:.2f} | SPS {sps}")

                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/mean_return_100", mean_ret, global_step)

                if args.wandb_project:
                    import wandb
                    wandb.log({
                        "td_loss": loss.item(),
                        "epsilon": epsilon,
                        "SPS": sps,
                        "mean_return_100": mean_ret,
                        "global_step": global_step,
                    })

        # Save checkpoint
        if global_step % train_cfg.get("save_interval_steps", 50_000) == 0 and global_step > 0:
            ckpt_path = output_dir / f"dqn_checkpoint_{global_step}.pt"
            torch.save({
                "global_step": global_step,
                "num_updates": num_updates,
                "online_state_dict": online_net.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

    # ------- Final save -------
    final_path = output_dir / "dqn_final_model.pt"
    torch.save(online_net.state_dict(), final_path)
    print(f"\nTraining complete! Final model → {final_path}")

    envs.close()
    writer.close()
    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
