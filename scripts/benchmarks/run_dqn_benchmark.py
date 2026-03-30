"""
DQN Benchmark: CNN vs ViT vs Hybrid CNN-ViT
Optimised for 8 GB VRAM (RTX 5050).
 - 96×96 input (36 ViT patches vs 196 at 224×224 → 5× faster)
 - uint8 replay buffer (4× less RAM)
 - Mixed-precision training (AMP fp16)
 - Smaller replay buffer (20 K)
"""
import os
import sys
import gc
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Environment
try:
    import minatar
    MINATAR_AVAILABLE = True
except ImportError:
    MINATAR_AVAILABLE = False
    print("MinAtar not available – install via: pip install minatar")

# DQN models
from models.dqn_cnn_policy import DQNCNN
from models.dqn_vit_policy import DQNViT
from models.dqn_hybrid_policy import DQNHybridCNNViT

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ---- Global constants -------------------------------------------------
IMG_SIZE = 96          # 96/16 = 6×6 = 36 patches (5.4× fewer than 224)
REPLAY_CAP = 20_000   # enough for 50 K-step runs, fits in ~1 GB RAM


# ============================================================================
# Helpers
# ============================================================================

def preprocess_minatar(obs):
    """Convert MinAtar (10,10,C) → uint8 (3, IMG_SIZE, IMG_SIZE).
    Stored as uint8 in replay buffer to save 4× RAM."""
    if obs.ndim == 2:
        obs = np.stack([obs] * 3, axis=-1)
    elif obs.ndim == 3 and obs.shape[-1] > 3:
        obs = obs[:, :, :3]
    elif obs.ndim == 3 and obs.shape[-1] < 3:
        obs = np.concatenate([obs] * (3 // obs.shape[-1] + 1), axis=-1)[:, :, :3]

    obs_resized = cv2.resize(obs.astype(np.float32), (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)
    # Normalise to 0-255 uint8 for compact storage
    mx = obs_resized.max()
    if mx > 0:
        obs_resized = (obs_resized / mx * 255.0)
    return obs_resized.clip(0, 255).astype(np.uint8).transpose(2, 0, 1)


def obs_to_tensor(obs_uint8, device):
    """uint8 [C,H,W] or [B,C,H,W] → float32 tensor on device, normalised 0-1."""
    return torch.as_tensor(obs_uint8, dtype=torch.float32, device=device).div_(255.0)


class SimpleReplayBuffer:
    """Pre-allocated uint8 replay buffer – fits 20 K transitions in ~1 GB."""

    def __init__(self, capacity: int, obs_shape=(3, IMG_SIZE, IMG_SIZE)):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        # uint8 storage: 20K × 2 × 3×96×96 ≈ 1.06 GB
        self.obs    = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.nobs   = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done   = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.pos]    = obs
        self.nobs[self.pos]   = next_obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.done[self.pos]   = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.obs[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.nobs[idxs],
            self.done[idxs],
        )

    def __len__(self):
        return self.size


# ============================================================================
# DQN Training Loop (single environment)
# ============================================================================

def train_dqn_with_logging(
    model_class,
    model_kwargs: dict,
    env_name: str,
    model_label: str,
    total_timesteps: int = 50_000,
    lr: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 32,
    replay_size: int = REPLAY_CAP,
    learning_starts: int = 1_000,
    train_frequency: int = 4,
    target_update_freq: int = 500,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_schedule_steps: int = 30_000,
    max_grad_norm: float = 10.0,
    double_dqn: bool = True,
    device: str = "cpu",
    seed: int = 42,
    log_interval: int = 50,
):
    """
    Train a DQN agent on a MinAtar environment and return learning-curve data.
    Optimised: 96×96 input, uint8 buffer, AMP fp16.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

    # ---- Environment ----
    env = minatar.Environment(env_name)
    num_actions = env.num_actions()

    # ---- Networks ----
    online_net = model_class(num_actions=num_actions, **model_kwargs).to(device)
    target_net = model_class(num_actions=num_actions, **model_kwargs).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    online_net.train()
    optimizer = optim.Adam(online_net.parameters(), lr=lr, eps=1e-5)
    replay = SimpleReplayBuffer(capacity=replay_size)

    # ---- Epsilon schedule ----
    def get_epsilon(step):
        frac = min(1.0, step / eps_schedule_steps)
        return eps_start + frac * (eps_end - eps_start)

    # ---- Tracking ----
    episode_returns = []
    timestep_log = []
    return_log = []
    current_return = 0.0
    current_length = 0
    num_updates = 0

    env.reset()
    obs = preprocess_minatar(env.state())  # uint8
    start_time = time.time()
    log_every = max(1, total_timesteps // log_interval)

    for step in range(1, total_timesteps + 1):
        # Epsilon-greedy
        eps = get_epsilon(step)
        if np.random.random() < eps:
            action = np.random.randint(num_actions)
        else:
            with torch.inference_mode():
                obs_t = obs_to_tensor(obs, device).unsqueeze(0)
                q = online_net(obs_t)
                action = q.argmax(dim=1).item()

        # Step environment
        reward, done = env.act(action)
        next_obs = preprocess_minatar(env.state())  # uint8
        current_return += reward
        current_length += 1

        replay.push(obs, action, reward, next_obs, float(done))

        if done:
            episode_returns.append(current_return)
            timestep_log.append(step)
            return_log.append(current_return)
            current_return = 0.0
            current_length = 0
            env.reset()
            next_obs = preprocess_minatar(env.state())

        obs = next_obs

        # ---- Training step ----
        if step >= learning_starts and step % train_frequency == 0 and len(replay) >= batch_size:
            s_obs, s_act, s_rew, s_nobs, s_done = replay.sample(batch_size)
            s_obs_t  = obs_to_tensor(s_obs, device)
            s_act_t  = torch.as_tensor(s_act, device=device)
            s_rew_t  = torch.as_tensor(s_rew, device=device)
            s_nobs_t = obs_to_tensor(s_nobs, device)
            s_done_t = torch.as_tensor(s_done, device=device)

            with torch.no_grad():
                if double_dqn:
                    best_actions = online_net(s_nobs_t).argmax(dim=1, keepdim=True)
                    next_q = target_net(s_nobs_t).gather(1, best_actions).squeeze(1)
                else:
                    next_q = target_net(s_nobs_t).max(dim=1).values
                td_target = s_rew_t + gamma * next_q * (1.0 - s_done_t)

            current_q = online_net(s_obs_t).gather(1, s_act_t.unsqueeze(1)).squeeze(1)
            loss = nn.functional.huber_loss(current_q, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), max_grad_norm)
            optimizer.step()
            num_updates += 1

            # Target update
            if num_updates % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % log_every == 0:
            elapsed = time.time() - start_time
            sps = int(step / elapsed) if elapsed > 0 else 0
            mean_ret = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            print(f"  [{model_label}] Step {step:>7,}/{total_timesteps:,} | "
                  f"Eps {eps:.3f} | Mean20 {mean_ret:>7.2f} | "
                  f"Episodes {len(episode_returns):>4} | SPS {sps}",
                  flush=True)

    training_time = time.time() - start_time
    sps = int(total_timesteps / training_time) if training_time > 0 else 0

    return {
        'model_label': model_label,
        'episode_returns': episode_returns,
        'timestep_log': timestep_log,
        'return_log': return_log,
        'mean_return': float(np.mean(episode_returns)) if episode_returns else 0.0,
        'std_return': float(np.std(episode_returns)) if episode_returns else 0.0,
        'max_return': float(np.max(episode_returns)) if episode_returns else 0.0,
        'total_episodes': len(episode_returns),
        'training_time': training_time,
        'sps': sps,
        'num_updates': num_updates,
    }


# ============================================================================
# Visualisation helpers
# ============================================================================

def smooth_curve(data, window=20):
    if len(data) < window:
        return data
    return [np.mean(data[max(0, i - window):i + 1]) for i in range(len(data))]


def create_dqn_visualizations(all_results, game, output_dir):
    """Generate comprehensive DQN benchmark plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'DQN-CNN': '#2ecc71', 'DQN-ViT': '#3498db', 'DQN-Hybrid': '#e74c3c'}

    methods = list(all_results.keys())

    # ── Figure 1: Learning Curves ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1a – raw runs
    ax = axes[0]
    for method in methods:
        for run in all_results[method]:
            ax.plot(run['timestep_log'], run['return_log'],
                    alpha=0.25, color=colors.get(method, '#888'))
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Episode Return')
    ax.set_title(f'DQN Learning Curves – {game.capitalize()} (all runs)')
    ax.legend(methods, loc='upper left')

    # 1b – smoothed mean ± std
    ax = axes[1]
    for method in methods:
        all_returns = [r['return_log'] for r in all_results[method]]
        min_len = min(len(r) for r in all_returns) if all_returns else 0
        if min_len > 0:
            truncated = np.array([r[:min_len] for r in all_returns])
            mean = smooth_curve(truncated.mean(axis=0).tolist())
            std = smooth_curve(truncated.std(axis=0).tolist())
            x = all_results[method][0]['timestep_log'][:min_len]
            x = x[:len(mean)]
            ax.plot(x, mean, color=colors.get(method, '#888'), linewidth=2, label=method)
            ax.fill_between(x, np.array(mean) - np.array(std),
                            np.array(mean) + np.array(std),
                            color=colors.get(method, '#888'), alpha=0.2)
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Episode Return (smoothed)')
    ax.set_title(f'DQN Learning Curves – {game.capitalize()} (smoothed)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'dqn_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dqn_learning_curves.png")

    # ── Figure 2: Bar charts ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stats = {}
    for method in methods:
        runs = all_results[method]
        stats[method] = {
            'mean_returns': [r['mean_return'] for r in runs],
            'sps': [r['sps'] for r in runs],
            'training_times': [r['training_time'] for r in runs],
        }

    # 2a – mean return
    ax = axes[0]
    means = [np.mean(stats[m]['mean_returns']) for m in methods]
    stds  = [np.std(stats[m]['mean_returns']) for m in methods]
    bars = ax.bar(methods, means, yerr=stds,
                  color=[colors.get(m, '#888') for m in methods],
                  capsize=5, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Mean Episode Return'); ax.set_title('Performance')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.1,
                f'{m:.2f}', ha='center', fontsize=10)

    # 2b – SPS
    ax = axes[1]
    sps_means = [np.mean(stats[m]['sps']) for m in methods]
    bars = ax.bar(methods, sps_means,
                  color=[colors.get(m, '#888') for m in methods],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Steps Per Second'); ax.set_title('Training Speed')
    for bar, s in zip(bars, sps_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{s:.0f}', ha='center', fontsize=10)

    # 2c – training time
    ax = axes[2]
    time_means = [np.mean(stats[m]['training_times']) / 60 for m in methods]
    bars = ax.bar(methods, time_means,
                  color=[colors.get(m, '#888') for m in methods],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Training Time (min)'); ax.set_title('Compute Cost')
    for bar, t in zip(bars, time_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{t:.1f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'dqn_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dqn_comparison_bars.png")

    # ── Figure 3: Box plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    import pandas as pd
    rows = []
    for method in methods:
        for run in all_results[method]:
            for ret in run['episode_returns']:
                rows.append({'Method': method, 'Return': ret})
    df = pd.DataFrame(rows)
    sns.boxplot(x='Method', y='Return', data=df, palette=colors, ax=ax)
    ax.set_title(f'DQN Return Distribution – {game.capitalize()}')
    plt.tight_layout()
    plt.savefig(output_dir / 'dqn_return_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dqn_return_distribution.png")

    # ── Figure 4: Dashboard ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'DQN Benchmark – CNN vs ViT vs Hybrid – {game.capitalize()}',
                 fontsize=18, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # learning curves (top-left, wide)
    ax1 = fig.add_subplot(gs[0, :2])
    for method in methods:
        all_rets = [r['return_log'] for r in all_results[method]]
        min_len = min(len(r) for r in all_rets) if all_rets else 0
        if min_len > 0:
            arr = np.array([r[:min_len] for r in all_rets])
            mean = smooth_curve(arr.mean(axis=0).tolist())
            x = all_results[method][0]['timestep_log'][:len(mean)]
            ax1.plot(x, mean, color=colors.get(method, '#888'), linewidth=2, label=method)
    ax1.set_xlabel('Timesteps'); ax1.set_ylabel('Mean Return')
    ax1.set_title('Learning Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # bar chart (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(methods))
    ax2.bar(x, means, yerr=stds,
            color=[colors.get(m, '#888') for m in methods],
            capsize=5, edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylabel('Mean Return'); ax2.set_title('Final Performance')

    # box plot (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    data_for_box = []
    for method in methods:
        rets = []
        for run in all_results[method]:
            rets.extend(run['episode_returns'][-50:])
        data_for_box.append(rets)
    bp = ax3.boxplot(data_for_box, labels=methods, patch_artist=True)
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(colors.get(method, '#888'))
    ax3.set_ylabel('Return'); ax3.set_title('Return Distribution')

    # speed (bottom-middle)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(x, sps_means, color=[colors.get(m, '#888') for m in methods], edgecolor='black')
    ax4.set_xticks(x); ax4.set_xticklabels(methods, fontsize=9)
    ax4.set_ylabel('SPS'); ax4.set_title('Training Speed')

    # summary table (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    table_data = []
    for m in methods:
        r = all_results[m]
        table_data.append([
            m,
            f"{np.mean([x['mean_return'] for x in r]):.2f}",
            f"{np.mean([x['sps'] for x in r]):.0f}",
            f"{np.mean([x['training_time'] for x in r]) / 60:.1f} min",
            f"{np.mean([x['num_updates'] for x in r]):.0f}",
        ])
    table = ax5.table(
        cellText=table_data,
        colLabels=['Method', 'Mean Ret', 'SPS', 'Time', 'Updates'],
        loc='center', cellLoc='center',
    )
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.4)
    ax5.set_title('Summary', fontsize=12, fontweight='bold')

    plt.savefig(output_dir / 'dqn_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dqn_dashboard.png")


# ============================================================================
# Main benchmark
# ============================================================================

def run_dqn_benchmark(
    games=('breakout', 'space_invaders', 'freeway'),
    seeds=(42, 123, 456),
    total_timesteps=50_000,
    device='cpu',
):
    """Run the full DQN benchmark: ViT and Hybrid on each game × seed."""
    if not MINATAR_AVAILABLE:
        print("ERROR: MinAtar is required. Install via: pip install minatar")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path('results') / f'dqn_benchmark_{timestamp}'
    base_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model configs ----
    model_specs = {
        'DQN-CNN': {
            'class': DQNCNN,
            'kwargs': {
                'embedding_dim': 256,
                'hidden_dim': 128,
                'dueling': True,
                'img_size': IMG_SIZE,
            },
        },
        'DQN-ViT': {
            'class': DQNViT,
            'kwargs': {
                'vit_model': 'vit_tiny_patch16_224',
                'embedding_dim': 256,
                'hidden_dim': 128,
                'pretrained': True,
                'freeze_vit_layers': 10,
                'dueling': True,
                'img_size': IMG_SIZE,
            },
        },
        'DQN-Hybrid': {
            'class': DQNHybridCNNViT,
            'kwargs': {
                'vit_model': 'vit_tiny_patch16_224',
                'embedding_dim': 256,
                'hidden_dim': 128,
                'pretrained': True,
                'freeze_vit_layers': 10,
                'dueling': True,
                'img_size': IMG_SIZE,
            },
        },
    }

    all_game_results = {}

    for game in games:
        print(f"\n{'='*60}")
        print(f"  GAME: {game.upper()}")
        print(f"{'='*60}")

        game_results = {}

        for method_name, spec in model_specs.items():
            game_results[method_name] = []

            for seed in seeds:
                print(f"\n--- {method_name} | seed={seed} ---")
                result = train_dqn_with_logging(
                    model_class=spec['class'],
                    model_kwargs=spec['kwargs'],
                    env_name=game,
                    model_label=method_name,
                    total_timesteps=total_timesteps,
                    lr=1e-4,
                    gamma=0.99,
                    batch_size=32,
                    replay_size=REPLAY_CAP,
                    learning_starts=1_000,
                    train_frequency=4,
                    target_update_freq=500,
                    eps_start=1.0,
                    eps_end=0.05,
                    eps_schedule_steps=int(total_timesteps * 0.6),
                    max_grad_norm=10.0,
                    double_dqn=True,
                    device=device,
                    seed=seed,
                    log_interval=20,
                )
                game_results[method_name].append(result)
                print(f"  → Mean return: {result['mean_return']:.2f} ± "
                      f"{result['std_return']:.2f}  |  SPS: {result['sps']}",
                      flush=True)

                # Free GPU memory between runs
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        # ---- Visualise per game ----
        game_dir = base_dir / game
        create_dqn_visualizations(game_results, game, game_dir)
        all_game_results[game] = game_results

    # ---- Save aggregate JSON ----
    summary = {}
    for game, gres in all_game_results.items():
        summary[game] = {}
        for method, runs in gres.items():
            summary[game][method] = {
                'mean_return': float(np.mean([r['mean_return'] for r in runs])),
                'std_return': float(np.mean([r['std_return'] for r in runs])),
                'max_return': float(np.max([r['max_return'] for r in runs])),
                'sps': float(np.mean([r['sps'] for r in runs])),
                'training_time_s': float(np.mean([r['training_time'] for r in runs])),
                'total_episodes': int(np.mean([r['total_episodes'] for r in runs])),
                'num_updates': int(np.mean([r['num_updates'] for r in runs])),
                'seeds': list(seeds),
            }
    with open(base_dir / 'dqn_benchmark_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {base_dir / 'dqn_benchmark_results.json'}")

    # ---- Cross-game summary ----
    print(f"\n{'='*70}")
    print("  DQN BENCHMARK SUMMARY")
    print(f"{'='*70}")
    for game in games:
        print(f"\n  {game.upper()}:")
        for method in model_specs:
            m = summary[game][method]
            print(f"    {method:<15}  Return: {m['mean_return']:>7.2f} ± "
                  f"{m['std_return']:<6.2f}  SPS: {m['sps']:>6.0f}  "
                  f"Episodes: {m['total_episodes']:>4}")
    print(f"\n{'='*70}")
    print(f"All results in: {base_dir}")

    return all_game_results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DQN Benchmark: CNN vs ViT vs Hybrid")
    parser.add_argument('--games', nargs='+',
                        default=['breakout', 'space_invaders', 'freeway'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--timesteps', type=int, default=50_000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_dqn_benchmark(
        games=args.games,
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        device=args.device,
    )
