"""
Full Benchmark: DQN vs C51 × CNN vs ViT vs Hybrid
Optimised for 8 GB VRAM (RTX 5050).
 - 96×96 input, uint8 replay buffer
 - DQN: Double Dueling DQN with Huber loss
 - C51: Categorical DQN with distributional projection loss
"""
import os, sys, gc, time, json
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

try:
    import minatar
    MINATAR_AVAILABLE = True
except ImportError:
    MINATAR_AVAILABLE = False
    print("MinAtar not available – pip install minatar")

# DQN models
from models.dqn_cnn_policy import DQNCNN
from models.dqn_vit_policy import DQNViT
from models.dqn_hybrid_policy import DQNHybridCNNViT
# C51 models
from models.c51_cnn_policy import C51CNN
from models.c51_vit_policy import C51ViT
from models.c51_hybrid_policy import C51HybridCNNViT

os.environ['PYTHONIOENCODING'] = 'utf-8'

IMG_SIZE = 96
REPLAY_CAP = 20_000
N_ATOMS = 51
V_MIN = -10.0
V_MAX = 10.0

# ============================================================================
# Helpers
# ============================================================================

def preprocess_minatar(obs):
    if obs.ndim == 2:
        obs = np.stack([obs] * 3, axis=-1)
    elif obs.ndim == 3 and obs.shape[-1] > 3:
        obs = obs[:, :, :3]
    elif obs.ndim == 3 and obs.shape[-1] < 3:
        obs = np.concatenate([obs] * (3 // obs.shape[-1] + 1), axis=-1)[:, :, :3]
    obs_resized = cv2.resize(obs.astype(np.float32), (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)
    mx = obs_resized.max()
    if mx > 0:
        obs_resized = (obs_resized / mx * 255.0)
    return obs_resized.clip(0, 255).astype(np.uint8).transpose(2, 0, 1)


def obs_to_tensor(obs_uint8, device):
    return torch.as_tensor(obs_uint8, dtype=torch.float32, device=device).div_(255.0)


class SimpleReplayBuffer:
    def __init__(self, capacity, obs_shape=(3, IMG_SIZE, IMG_SIZE)):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
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

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (self.obs[idxs], self.action[idxs], self.reward[idxs],
                self.nobs[idxs], self.done[idxs])

    def __len__(self):
        return self.size


# ============================================================================
# C51 projection (distributional Bellman update)
# ============================================================================

def c51_projection(next_dist, rewards, dones, gamma, support, n_atoms, v_min, v_max, delta_z):
    """
    Categorical projection for distributional RL.
    next_dist : [B, N_ATOMS]  – atom probabilities of next_state best action
    rewards   : [B]
    dones     : [B]
    Returns   : [B, N_ATOMS]  – projected target distribution
    """
    batch_size = rewards.shape[0]
    # Tz = r + gamma * z  (clipped)
    Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    Tz = Tz.clamp(v_min, v_max)

    # Map to atom indices
    b = (Tz - v_min) / delta_z  # [B, N_ATOMS]
    l = b.floor().long()
    u = b.ceil().long()
    # Clamp indices
    l = l.clamp(0, n_atoms - 1)
    u = u.clamp(0, n_atoms - 1)

    # Distribute probability
    target_dist = torch.zeros_like(next_dist)  # [B, N_ATOMS]
    offset = torch.arange(batch_size, device=rewards.device).unsqueeze(1) * n_atoms

    target_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                     (next_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                     (next_dist * (b - l.float())).view(-1))
    return target_dist


# ============================================================================
# Unified training loop: DQN or C51
# ============================================================================

def train_agent(
    model_class, model_kwargs, env_name, model_label,
    algo='dqn',  # 'dqn' or 'c51'
    total_timesteps=50_000, lr=1e-4, gamma=0.99, batch_size=32,
    replay_size=REPLAY_CAP, learning_starts=1_000, train_frequency=4,
    target_update_freq=500, eps_start=1.0, eps_end=0.05,
    eps_schedule_steps=30_000, max_grad_norm=10.0, double_dqn=True,
    device='cpu', seed=42, log_interval=20,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

    env = minatar.Environment(env_name)
    num_actions = env.num_actions()

    online_net = model_class(num_actions=num_actions, **model_kwargs).to(device)
    target_net = model_class(num_actions=num_actions, **model_kwargs).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    online_net.train()

    optimizer = optim.Adam(online_net.parameters(), lr=lr, eps=1e-5)
    replay = SimpleReplayBuffer(capacity=replay_size)

    # C51 support
    is_c51 = (algo == 'c51')
    if is_c51:
        n_atoms = online_net.n_atoms
        v_min = online_net.v_min
        v_max = online_net.v_max
        delta_z = online_net.delta_z
        support = online_net.support  # on device after .to(device)

    def get_epsilon(step):
        frac = min(1.0, step / eps_schedule_steps)
        return eps_start + frac * (eps_end - eps_start)

    episode_returns = []
    timestep_log = []
    return_log = []
    current_return = 0.0
    num_updates = 0

    env.reset()
    obs = preprocess_minatar(env.state())
    start_time = time.time()
    log_every = max(1, total_timesteps // log_interval)

    for step in range(1, total_timesteps + 1):
        eps = get_epsilon(step)
        if np.random.random() < eps:
            action = np.random.randint(num_actions)
        else:
            with torch.inference_mode():
                obs_t = obs_to_tensor(obs, device).unsqueeze(0)
                q = online_net(obs_t)  # both DQN and C51 .forward() return Q values
                action = q.argmax(dim=1).item()

        reward, done = env.act(action)
        next_obs = preprocess_minatar(env.state())
        current_return += reward
        replay.push(obs, action, reward, next_obs, float(done))

        if done:
            episode_returns.append(current_return)
            timestep_log.append(step)
            return_log.append(current_return)
            current_return = 0.0
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

            if is_c51:
                # ---- C51 distributional loss ----
                # Current distribution
                current_dist = online_net.dist(s_obs_t)  # [B, A, N]
                current_dist = current_dist[
                    torch.arange(batch_size, device=device), s_act_t
                ]  # [B, N]

                with torch.no_grad():
                    # Greedy action from online net (Double DQN style)
                    if double_dqn:
                        next_q = online_net(s_nobs_t)  # [B, A]
                        best_a = next_q.argmax(dim=1)
                    else:
                        next_q = target_net(s_nobs_t)
                        best_a = next_q.argmax(dim=1)

                    next_dist = target_net.dist(s_nobs_t)  # [B, A, N]
                    next_dist = next_dist[
                        torch.arange(batch_size, device=device), best_a
                    ]  # [B, N]

                    target_dist = c51_projection(
                        next_dist, s_rew_t, s_done_t, gamma,
                        support, n_atoms, v_min, v_max, delta_z,
                    )

                # Cross-entropy loss
                loss = -(target_dist * (current_dist + 1e-8).log()).sum(dim=-1).mean()
            else:
                # ---- DQN Huber loss ----
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

            if num_updates % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % log_every == 0:
            elapsed = time.time() - start_time
            sps = int(step / elapsed) if elapsed > 0 else 0
            mean_ret = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            print(f"  [{model_label}] Step {step:>7,}/{total_timesteps:,} | "
                  f"Eps {eps:.3f} | Mean20 {mean_ret:>7.2f} | "
                  f"Episodes {len(episode_returns):>4} | SPS {sps}", flush=True)

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
# Visualisation
# ============================================================================

def smooth_curve(data, window=20):
    if len(data) < window:
        return data
    return [np.mean(data[max(0, i - window):i + 1]) for i in range(len(data))]


def create_visualizations(all_results, game, output_dir, title_prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    COLORS = {
        # DQN
        'DQN-CNN':    '#2ecc71', 'DQN-ViT':    '#3498db', 'DQN-Hybrid':    '#e74c3c',
        # C51
        'C51-CNN':    '#27ae60', 'C51-ViT':    '#2980b9', 'C51-Hybrid':    '#c0392b',
    }
    methods = list(all_results.keys())
    colors = {m: COLORS.get(m, '#888888') for m in methods}

    # ── Learning curves ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    for method in methods:
        for run in all_results[method]:
            ax.plot(run['timestep_log'], run['return_log'],
                    alpha=0.2, color=colors[method])
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Episode Return')
    ax.set_title(f'{title_prefix}{game.capitalize()} – All Runs')
    ax.legend(methods, loc='upper left', fontsize=7)

    ax = axes[1]
    for method in methods:
        all_rets = [r['return_log'] for r in all_results[method]]
        min_len = min(len(r) for r in all_rets) if all_rets else 0
        if min_len > 0:
            trunc = np.array([r[:min_len] for r in all_rets])
            mean = smooth_curve(trunc.mean(axis=0).tolist())
            std = smooth_curve(trunc.std(axis=0).tolist())
            x = all_results[method][0]['timestep_log'][:min_len][:len(mean)]
            ax.plot(x, mean, color=colors[method], linewidth=2, label=method)
            ax.fill_between(x, np.array(mean) - np.array(std),
                            np.array(mean) + np.array(std),
                            color=colors[method], alpha=0.15)
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Episode Return (smoothed)')
    ax.set_title(f'{title_prefix}{game.capitalize()} – Smoothed Mean±Std')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved: learning_curves.png")

    # ── Bar charts ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    stats = {}
    for m in methods:
        runs = all_results[m]
        stats[m] = {
            'means': [r['mean_return'] for r in runs],
            'sps': [r['sps'] for r in runs],
            'times': [r['training_time'] for r in runs],
        }

    ax = axes[0]
    means_ = [np.mean(stats[m]['means']) for m in methods]
    stds_  = [np.std(stats[m]['means']) for m in methods]
    bars = ax.bar(methods, means_, yerr=stds_,
                  color=[colors[m] for m in methods], capsize=4, edgecolor='black')
    ax.set_ylabel('Mean Return'); ax.set_title('Performance')
    ax.tick_params(axis='x', rotation=25)
    for b, mv, sv in zip(bars, means_, stds_):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sv + 0.05,
                f'{mv:.2f}', ha='center', fontsize=8)

    ax = axes[1]
    sps_m = [np.mean(stats[m]['sps']) for m in methods]
    ax.bar(methods, sps_m, color=[colors[m] for m in methods], edgecolor='black')
    ax.set_ylabel('Steps/sec'); ax.set_title('Throughput')
    ax.tick_params(axis='x', rotation=25)

    ax = axes[2]
    times_m = [np.mean(stats[m]['times'])/60 for m in methods]
    ax.bar(methods, times_m, color=[colors[m] for m in methods], edgecolor='black')
    ax.set_ylabel('Minutes'); ax.set_title('Training Time')
    ax.tick_params(axis='x', rotation=25)

    plt.tight_layout()
    plt.savefig(output_dir / 'bar_charts.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved: bar_charts.png")

    # ── Dashboard ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{title_prefix}{game.capitalize()} – Dashboard', fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Smoothed curves
    ax1 = fig.add_subplot(gs[0, :2])
    for method in methods:
        all_rets = [r['return_log'] for r in all_results[method]]
        min_len = min(len(r) for r in all_rets) if all_rets else 0
        if min_len > 0:
            trunc = np.array([r[:min_len] for r in all_rets])
            mean = smooth_curve(trunc.mean(axis=0).tolist())
            std = smooth_curve(trunc.std(axis=0).tolist())
            x = all_results[method][0]['timestep_log'][:min_len][:len(mean)]
            ax1.plot(x, mean, color=colors[method], linewidth=2, label=method)
            ax1.fill_between(x, np.array(mean)-np.array(std),
                             np.array(mean)+np.array(std),
                             color=colors[method], alpha=0.12)
    ax1.set_xlabel('Timesteps'); ax1.set_ylabel('Return')
    ax1.set_title('Learning Curves (mean±std)'); ax1.legend(fontsize=7)

    # Performance bars
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(methods, means_, yerr=stds_,
            color=[colors[m] for m in methods], capsize=3, edgecolor='black')
    ax2.set_title('Mean Return'); ax2.tick_params(axis='x', rotation=30, labelsize=7)

    # SPS bars
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(methods, sps_m, color=[colors[m] for m in methods], edgecolor='black')
    ax3.set_title('Throughput (SPS)'); ax3.tick_params(axis='x', rotation=30, labelsize=7)

    # Box plot
    ax4 = fig.add_subplot(gs[1, 1])
    data_box = [stats[m]['means'] for m in methods]
    bp = ax4.boxplot(data_box, tick_labels=methods, patch_artist=True)
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(colors[m])
    ax4.set_title('Return Distribution'); ax4.tick_params(axis='x', rotation=30, labelsize=7)

    # Summary table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    tdata = []
    for m in methods:
        r = all_results[m]
        tdata.append([
            m,
            f"{np.mean([x['mean_return'] for x in r]):.2f}",
            f"{np.mean([x['sps'] for x in r]):.0f}",
            f"{np.mean([x['training_time'] for x in r])/60:.1f}m",
        ])
    table = ax5.table(cellText=tdata, colLabels=['Method','Return','SPS','Time'],
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
    ax5.set_title('Summary', fontsize=11, fontweight='bold')

    plt.savefig(output_dir / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved: dashboard.png")


# ============================================================================
# Main benchmark
# ============================================================================

def run_benchmark(
    games=('breakout', 'space_invaders', 'freeway'),
    seeds=(42, 123, 456),
    total_timesteps=50_000,
    device='cpu',
    algos=('dqn', 'c51'),
):
    if not MINATAR_AVAILABLE:
        print("ERROR: MinAtar required"); return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path('results') / f'full_benchmark_{timestamp}'
    base_dir.mkdir(parents=True, exist_ok=True)

    shared_kwargs = {
        'embedding_dim': 256,
        'hidden_dim': 128,
        'dueling': True,
        'img_size': IMG_SIZE,
    }
    vit_kwargs = {
        **shared_kwargs,
        'vit_model': 'vit_tiny_patch16_224',
        'pretrained': True,
        'freeze_vit_layers': 10,
    }
    hybrid_kwargs = {
        **vit_kwargs,
        'vit_grad_scale': 0.1,
    }

    # All model specs: (label, algo, class, kwargs)
    model_specs = []
    if 'dqn' in algos:
        model_specs += [
            ('DQN-CNN',    'dqn', DQNCNN,          shared_kwargs),
            ('DQN-ViT',    'dqn', DQNViT,          vit_kwargs),
            ('DQN-Hybrid', 'dqn', DQNHybridCNNViT, hybrid_kwargs),
        ]
    if 'c51' in algos:
        c51_extra = {'n_atoms': N_ATOMS, 'v_min': V_MIN, 'v_max': V_MAX}
        model_specs += [
            ('C51-CNN',    'c51', C51CNN,          {**shared_kwargs, **c51_extra}),
            ('C51-ViT',    'c51', C51ViT,          {**vit_kwargs, **c51_extra}),
            ('C51-Hybrid', 'c51', C51HybridCNNViT, {**hybrid_kwargs, **c51_extra}),
        ]

    all_game_results = {}

    for game in games:
        print(f"\n{'='*60}")
        print(f"  GAME: {game.upper()}")
        print(f"{'='*60}")

        game_results = {}

        for label, algo, cls, kwargs in model_specs:
            game_results[label] = []

            for seed in seeds:
                print(f"\n--- {label} | seed={seed} ---")
                result = train_agent(
                    model_class=cls,
                    model_kwargs=kwargs,
                    env_name=game,
                    model_label=label,
                    algo=algo,
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
                game_results[label].append(result)
                print(f"  → {label} seed={seed}: mean={result['mean_return']:.2f} ± "
                      f"{result['std_return']:.2f}  SPS={result['sps']}", flush=True)

                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        # Visualise per game
        game_dir = base_dir / game
        create_visualizations(game_results, game, game_dir, title_prefix="DQN vs C51 – ")
        all_game_results[game] = game_results

    # ---- Aggregate JSON ----
    summary = {}
    for game, gres in all_game_results.items():
        summary[game] = {}
        for method, runs in gres.items():
            summary[game][method] = {
                'mean_return': float(np.mean([r['mean_return'] for r in runs])),
                'std_return':  float(np.mean([r['std_return'] for r in runs])),
                'max_return':  float(np.max([r['max_return'] for r in runs])),
                'sps':         float(np.mean([r['sps'] for r in runs])),
                'training_s':  float(np.mean([r['training_time'] for r in runs])),
                'episodes':    int(np.mean([r['total_episodes'] for r in runs])),
                'updates':     int(np.mean([r['num_updates'] for r in runs])),
                'seeds':       list(seeds),
            }
    with open(base_dir / 'benchmark_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print("  FULL BENCHMARK SUMMARY  (DQN vs C51 × CNN vs ViT vs Hybrid)")
    print(f"{'='*80}")
    for game in games:
        print(f"\n  {game.upper()}:")
        for label, _, _, _ in model_specs:
            m = summary[game][label]
            print(f"    {label:<15}  Return: {m['mean_return']:>7.2f} ± "
                  f"{m['std_return']:<6.2f}  SPS: {m['sps']:>6.0f}  "
                  f"Eps: {m['episodes']:>4}")
    print(f"\n{'='*80}")
    print(f"All results in: {base_dir}")

    return all_game_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full Benchmark: DQN vs C51 × CNN/ViT/Hybrid")
    parser.add_argument('--games', nargs='+', default=['breakout', 'space_invaders', 'freeway'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--timesteps', type=int, default=50_000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--algos', nargs='+', default=['dqn', 'c51'],
                        choices=['dqn', 'c51'])
    args = parser.parse_args()

    run_benchmark(
        games=args.games,
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        device=args.device,
        algos=args.algos,
    )
