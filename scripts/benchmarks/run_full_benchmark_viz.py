"""
Full Benchmark with Visualization
Compares CNN, ViT, and ViT+Text across multiple seeds and generates plots
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import time
import json
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

from minatar import Environment

from models.vit_policy import ActorCriticViT
from models.vit_text_policy import ActorCriticViTText
from models.cnn_policy import SmallCNNPolicy
from utils.text_history import HistoryBuffer


class MinAtarWrapper:
    """Wraps MinAtar environment"""
    
    def __init__(self, game='breakout', seed=42):
        self.env = Environment(game)
        self.env.seed(seed)
        self.game = game
        self.action_space_n = self.env.num_actions()
        
    def reset(self):
        self.env.reset()
        state = self.env.state()
        return self._process_state(state), {}
    
    def step(self, action):
        reward, done = self.env.act(action)
        state = self.env.state()
        return self._process_state(state), reward, done, False, {}
    
    def _process_state(self, state):
        import cv2
        if state.ndim == 3:
            h, w, c = state.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(min(c, 3)):
                rgb[:, :, i] = state[:, :, i].astype(np.float32)
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = (state * 255).clip(0, 255).astype(np.uint8)
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        rgb_resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
        return rgb_resized.transpose(2, 0, 1)
    
    def close(self):
        pass


def train_with_logging(
    policy,
    env,
    policy_name: str,
    use_text: bool = False,
    total_timesteps: int = 50000,
    num_steps: int = 128,
    num_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    device: str = 'cpu',
    log_interval: int = 10,
):
    """Train and log learning curve data"""
    
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    
    if use_text:
        history = HistoryBuffer(history_length=5, template_style='detailed')
    
    obs_storage = torch.zeros((num_steps, 3, 224, 224)).to(device)
    actions_storage = torch.zeros(num_steps).to(device)
    logprobs_storage = torch.zeros(num_steps).to(device)
    rewards_storage = torch.zeros(num_steps).to(device)
    dones_storage = torch.zeros(num_steps).to(device)
    values_storage = torch.zeros(num_steps).to(device)
    
    if use_text:
        text_storage = [None] * num_steps
    
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs).to(device) / 255.0
    done = False
    
    episode_returns = []
    episode_lengths = []
    current_return = 0
    current_length = 0
    
    # Logging data
    timestep_log = []
    return_log = []
    
    num_updates = total_timesteps // num_steps
    start_time = time.time()
    global_step = 0
    
    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            global_step += 1
            obs_storage[step] = obs
            dones_storage[step] = done
            
            with torch.no_grad():
                if use_text:
                    text = history.to_text()
                    text_storage[step] = text
                    action, logprob, _, value = policy.get_action_and_value(
                        obs.unsqueeze(0), [text]
                    )
                else:
                    action, logprob, _, value = policy.get_action_and_value(obs.unsqueeze(0))
            
            actions_storage[step] = action
            logprobs_storage[step] = logprob
            values_storage[step] = value.flatten()
            
            next_obs, reward, done, truncated, _ = env.step(action.item())
            rewards_storage[step] = reward
            
            current_return += reward
            current_length += 1
            
            if use_text:
                history.add(action=action.item(), reward=reward, done=done)
            
            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                
                # Log for learning curve
                timestep_log.append(global_step)
                return_log.append(current_return)
                
                current_return = 0
                current_length = 0
                next_obs, _ = env.reset()
                if use_text:
                    history.reset()
            
            obs = torch.FloatTensor(next_obs).to(device) / 255.0
        
        # Bootstrap
        with torch.no_grad():
            if use_text:
                _, _, _, next_value = policy.get_action_and_value(obs.unsqueeze(0), [history.to_text()])
            else:
                _, _, _, next_value = policy.get_action_and_value(obs.unsqueeze(0))
            next_value = next_value.flatten()
        
        # GAE
        advantages = torch.zeros_like(rewards_storage)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_storage[t + 1]
                nextvalues = values_storage[t + 1]
            delta = rewards_storage[t] + gamma * nextvalues * nextnonterminal - values_storage[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values_storage
        
        b_inds = np.arange(num_steps)
        for epoch in range(num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]
                
                if use_text:
                    mb_texts = [text_storage[i] for i in mb_inds]
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        obs_storage[mb_inds], mb_texts, actions_storage[mb_inds].long()
                    )
                else:
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        obs_storage[mb_inds], action=actions_storage[mb_inds].long()
                    )
                
                logratio = newlogprob - logprobs_storage[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((newvalue.flatten() - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        
        if update % log_interval == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            if episode_returns:
                mean_ret = np.mean(episode_returns[-10:])
                print(f"  [{policy_name}] {update}/{num_updates} | SPS: {sps} | Return: {mean_ret:.2f} | Episodes: {len(episode_returns)}")
    
    training_time = time.time() - start_time
    
    return {
        'policy_name': policy_name,
        'episode_returns': episode_returns,
        'timestep_log': timestep_log,
        'return_log': return_log,
        'mean_return': float(np.mean(episode_returns)) if episode_returns else 0,
        'std_return': float(np.std(episode_returns)) if episode_returns else 0,
        'max_return': float(np.max(episode_returns)) if episode_returns else 0,
        'total_episodes': len(episode_returns),
        'training_time': training_time,
        'sps': int(total_timesteps / training_time),
    }


def smooth_curve(data, window=50):
    """Smooth data with moving average"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2)
        smoothed.append(np.mean(data[start:end]))
    return smoothed


def create_visualizations(all_results, game, output_dir):
    """Create comprehensive visualizations"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'CNN': '#2ecc71', 'ViT': '#3498db', 'ViT+Text': '#9b59b6'}
    
    # ============================================
    # Figure 1: Learning Curves
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1a: All runs
    ax1 = axes[0]
    for method in ['CNN', 'ViT', 'ViT+Text']:
        method_results = [r for r in all_results if r['policy_name'] == method]
        for i, r in enumerate(method_results):
            if r['timestep_log'] and r['return_log']:
                alpha = 0.3 if len(method_results) > 1 else 1.0
                ax1.plot(r['timestep_log'], r['return_log'], 
                        color=colors[method], alpha=alpha, linewidth=0.8)
    
    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Episode Return', fontsize=12)
    ax1.set_title(f'Learning Curves - {game.capitalize()} (All Runs)', fontsize=14)
    ax1.legend(['CNN', 'ViT', 'ViT+Text'], loc='upper left')
    
    # Plot 1b: Smoothed averages
    ax2 = axes[1]
    for method in ['CNN', 'ViT', 'ViT+Text']:
        method_results = [r for r in all_results if r['policy_name'] == method]
        
        # Aggregate returns by binning timesteps
        all_timesteps = []
        all_returns = []
        for r in method_results:
            all_timesteps.extend(r['timestep_log'])
            all_returns.extend(r['return_log'])
        
        if all_timesteps:
            # Sort by timestep
            sorted_pairs = sorted(zip(all_timesteps, all_returns))
            timesteps = [p[0] for p in sorted_pairs]
            returns = [p[1] for p in sorted_pairs]
            
            # Smooth
            smoothed = smooth_curve(returns, window=100)
            ax2.plot(timesteps, smoothed, color=colors[method], 
                    linewidth=2.5, label=method)
    
    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Episode Return (Smoothed)', fontsize=12)
    ax2.set_title(f'Learning Curves - {game.capitalize()} (Smoothed)', fontsize=14)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'learning_curves.png'}")
    
    # ============================================
    # Figure 2: Bar Chart Comparison
    # ============================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['CNN', 'ViT', 'ViT+Text']
    
    # Aggregate stats
    stats = {}
    for method in methods:
        method_results = [r for r in all_results if r['policy_name'] == method]
        if method_results:
            stats[method] = {
                'mean_returns': [r['mean_return'] for r in method_results],
                'max_returns': [r['max_return'] for r in method_results],
                'training_times': [r['training_time'] for r in method_results],
                'sps': [r['sps'] for r in method_results],
            }
    
    # Plot 2a: Mean Return
    ax1 = axes[0]
    means = [np.mean(stats[m]['mean_returns']) for m in methods]
    stds = [np.std(stats[m]['mean_returns']) for m in methods]
    bars = ax1.bar(methods, means, yerr=stds, color=[colors[m] for m in methods],
                   capsize=5, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean Episode Return', fontsize=12)
    ax1.set_title('Performance Comparison', fontsize=14)
    ax1.set_ylim(0, max(means) * 1.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2b: Training Speed (SPS)
    ax2 = axes[1]
    sps_means = [np.mean(stats[m]['sps']) for m in methods]
    bars = ax2.bar(methods, sps_means, color=[colors[m] for m in methods],
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Steps Per Second (SPS)', fontsize=12)
    ax2.set_title('Training Speed', fontsize=14)
    ax2.set_yscale('log')
    
    for bar, sps in zip(bars, sps_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{sps}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2c: Training Time
    ax3 = axes[2]
    time_means = [np.mean(stats[m]['training_times']) / 60 for m in methods]  # minutes
    bars = ax3.bar(methods, time_means, color=[colors[m] for m in methods],
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Training Time (minutes)', fontsize=12)
    ax3.set_title('Compute Cost', fontsize=14)
    
    for bar, t in zip(bars, time_means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{t:.1f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'comparison_bars.png'}")
    
    # ============================================
    # Figure 3: Box Plot
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    labels = []
    for method in methods:
        method_results = [r for r in all_results if r['policy_name'] == method]
        for r in method_results:
            data.extend(r['episode_returns'])
            labels.extend([method] * len(r['episode_returns']))
    
    import pandas as pd
    df = pd.DataFrame({'Method': labels, 'Return': data})
    
    sns.boxplot(x='Method', y='Return', data=df, palette=colors, ax=ax)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title(f'Return Distribution - {game.capitalize()}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'return_distribution.png'}")
    
    # ============================================
    # Figure 4: Summary Dashboard
    # ============================================
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle(f'ViT vs CNN vs Multimodal RL Benchmark - {game.capitalize()}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Learning curves
    ax1 = fig.add_subplot(gs[0, :2])
    for method in methods:
        method_results = [r for r in all_results if r['policy_name'] == method]
        all_timesteps = []
        all_returns = []
        for r in method_results:
            all_timesteps.extend(r['timestep_log'])
            all_returns.extend(r['return_log'])
        if all_timesteps:
            sorted_pairs = sorted(zip(all_timesteps, all_returns))
            timesteps = [p[0] for p in sorted_pairs]
            returns = [p[1] for p in sorted_pairs]
            smoothed = smooth_curve(returns, window=100)
            ax1.plot(timesteps, smoothed, color=colors[method], linewidth=2.5, label=method)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Return (Smoothed)')
    ax1.set_title('Learning Curves')
    ax1.legend()
    
    # Stats table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    table_data = []
    for method in methods:
        method_results = [r for r in all_results if r['policy_name'] == method]
        if method_results:
            mean_ret = np.mean([r['mean_return'] for r in method_results])
            std_ret = np.std([r['mean_return'] for r in method_results])
            max_ret = np.max([r['max_return'] for r in method_results])
            avg_sps = np.mean([r['sps'] for r in method_results])
            table_data.append([method, f'{mean_ret:.2f}+/-{std_ret:.2f}', f'{max_ret:.1f}', f'{avg_sps:.0f}'])
    
    table = ax2.table(cellText=table_data,
                     colLabels=['Method', 'Mean Return', 'Max', 'SPS'],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax2.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Bar chart - performance
    ax3 = fig.add_subplot(gs[1, 0])
    means = [np.mean(stats[m]['mean_returns']) for m in methods]
    stds = [np.std(stats[m]['mean_returns']) for m in methods]
    ax3.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], capsize=5)
    ax3.set_ylabel('Mean Return')
    ax3.set_title('Performance')
    
    # Bar chart - speed
    ax4 = fig.add_subplot(gs[1, 1])
    sps_means = [np.mean(stats[m]['sps']) for m in methods]
    ax4.bar(methods, sps_means, color=[colors[m] for m in methods])
    ax4.set_ylabel('Steps/Second')
    ax4.set_title('Training Speed')
    ax4.set_yscale('log')
    
    # Box plot
    ax5 = fig.add_subplot(gs[1, 2])
    sns.boxplot(x='Method', y='Return', data=df, palette=colors, ax=ax5)
    ax5.set_title('Return Distribution')
    
    plt.savefig(output_dir / 'benchmark_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'benchmark_dashboard.png'}")
    
    return output_dir


def run_full_benchmark(game='breakout', seeds=[42, 123], timesteps=30000):
    """Run full benchmark with multiple seeds"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("FULL BENCHMARK WITH VISUALIZATION")
    print("="*70)
    print(f"Game: {game}")
    print(f"Seeds: {seeds}")
    print(f"Timesteps per run: {timesteps:,}")
    print(f"Device: {device}")
    print("="*70)
    
    all_results = []
    
    env = MinAtarWrapper(game=game, seed=42)
    num_actions = env.action_space_n
    env.close()
    
    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED: {seed}")
        print(f"{'#'*70}")
        
        # CNN
        print(f"\n--- Training CNN (seed={seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = MinAtarWrapper(game=game, seed=seed)
        cnn_policy = SmallCNNPolicy(num_actions=num_actions, hidden_dim=128).to(device)
        
        cnn_results = train_with_logging(
            policy=cnn_policy, env=env, policy_name="CNN",
            use_text=False, total_timesteps=timesteps, device=device
        )
        cnn_results['seed'] = seed
        cnn_results['parameters'] = sum(p.numel() for p in cnn_policy.parameters())
        all_results.append(cnn_results)
        env.close()
        del cnn_policy
        
        # ViT
        print(f"\n--- Training ViT (seed={seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = MinAtarWrapper(game=game, seed=seed)
        vit_policy = ActorCriticViT(
            num_actions=num_actions, vit_model="vit_tiny_patch16_224",
            embedding_dim=192, hidden_dim=128, pretrained=True, freeze_vit_layers=8
        ).to(device)
        
        vit_results = train_with_logging(
            policy=vit_policy, env=env, policy_name="ViT",
            use_text=False, total_timesteps=timesteps, device=device
        )
        vit_results['seed'] = seed
        vit_results['parameters'] = sum(p.numel() for p in vit_policy.parameters())
        all_results.append(vit_results)
        env.close()
        del vit_policy
        
        # ViT+Text
        print(f"\n--- Training ViT+Text (seed={seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = MinAtarWrapper(game=game, seed=seed)
        mm_policy = ActorCriticViTText(
            num_actions=num_actions, vit_model="vit_tiny_patch16_224",
            text_model="distilbert-base-uncased", embedding_dim=192, hidden_dim=128,
            fusion_type="concat", pretrained=True, freeze_vit_layers=8, freeze_text=True
        ).to(device)
        
        mm_results = train_with_logging(
            policy=mm_policy, env=env, policy_name="ViT+Text",
            use_text=True, total_timesteps=timesteps, device=device
        )
        mm_results['seed'] = seed
        mm_results['parameters'] = sum(p.numel() for p in mm_policy.parameters())
        all_results.append(mm_results)
        env.close()
        del mm_policy
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = create_visualizations(all_results, game, f'results/benchmark_{game}')
    
    # Save raw results
    results_file = output_dir / 'benchmark_results.json'
    
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, list):
                jr[k] = [convert_for_json(x) for x in v]
            else:
                jr[k] = convert_for_json(v)
        json_results.append(jr)
    
    with open(results_file, 'w') as f:
        json.dump({
            'game': game,
            'seeds': seeds,
            'timesteps': timesteps,
            'results': json_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Saved: {results_file}")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL BENCHMARK RESULTS")
    print("="*70)
    
    print(f"\n{'Method':<12} {'Mean Return':>14} {'Max':>8} {'SPS':>8} {'Time':>10}")
    print("-"*60)
    
    for method in ['CNN', 'ViT', 'ViT+Text']:
        method_results = [r for r in all_results if r['policy_name'] == method]
        if method_results:
            mean_ret = np.mean([r['mean_return'] for r in method_results])
            std_ret = np.std([r['mean_return'] for r in method_results])
            max_ret = np.max([r['max_return'] for r in method_results])
            avg_sps = np.mean([r['sps'] for r in method_results])
            avg_time = np.mean([r['training_time'] for r in method_results])
            print(f"{method:<12} {mean_ret:>7.2f} +/- {std_ret:<4.2f} {max_ret:>8.1f} {avg_sps:>8.0f} {avg_time:>9.0f}s")
    
    print("-"*60)
    print(f"\nVisualizations saved to: {output_dir}/")
    print("  - learning_curves.png")
    print("  - comparison_bars.png")
    print("  - return_distribution.png")
    print("  - benchmark_dashboard.png")
    print("="*70)
    
    return all_results, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    parser.add_argument('--timesteps', type=int, default=30000)
    args = parser.parse_args()
    
    results, output_dir = run_full_benchmark(
        game=args.game,
        seeds=args.seeds,
        timesteps=args.timesteps,
    )
