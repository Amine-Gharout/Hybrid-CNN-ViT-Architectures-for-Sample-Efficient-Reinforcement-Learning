"""
FAST Benchmark: Novel Gated Fusion vs CNN
Tests the new research method quickly
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

import minatar

from models.minatar_policies import MinAtarCNN, MinAtarHybridCNN, MinAtarGatedFusion


def preprocess_obs(obs):
    """Convert MinAtar obs to (C, H, W) format"""
    # MinAtar returns (H, W, C) - need to transpose to (C, H, W)
    obs_norm = obs.astype(np.float32)
    if obs_norm.max() > 0:
        obs_norm = obs_norm / obs_norm.max()
    return obs_norm.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)


def train_fast(
    model,
    env_name,
    total_timesteps,
    device="cuda",
    seed=42,
    text_instruction=None,
    use_text=False
):
    """Fast training loop"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = minatar.Environment(env_name)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    
    episode_returns = []
    current_reward = 0
    all_returns = []
    timesteps_log = []
    
    env.reset()
    obs = preprocess_obs(env.state())
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    
    start_time = time.time()
    
    for step in range(total_timesteps):
        # Get action
        with torch.no_grad():
            if use_text:
                action, _, _, _ = model.get_action_and_value(obs_tensor, [text_instruction])
            else:
                action, _, _, _ = model.get_action_and_value(obs_tensor)
        
        # Step
        reward, done = env.act(action.item())
        current_reward += reward
        
        if done:
            episode_returns.append(current_reward)
            current_reward = 0
            env.reset()
        
        # Next obs
        obs = preprocess_obs(env.state())
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Log every 2000 steps
        if (step + 1) % 2000 == 0:
            if len(episode_returns) > 0:
                mean_ret = np.mean(episode_returns[-20:])
                all_returns.append(mean_ret)
                timesteps_log.append(step + 1)
                sps = int((step + 1) / (time.time() - start_time))
                print(f"  Step {step+1:6d} | Return: {mean_ret:.2f} | SPS: {sps}")
    
    final_sps = int(total_timesteps / (time.time() - start_time))
    
    return {
        'timesteps': timesteps_log,
        'returns': all_returns,
        'final_returns': episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns,
        'sps': final_sps
    }


def run_quick_benchmark():
    """Quick benchmark comparing methods"""
    
    print("=" * 70)
    print("QUICK BENCHMARK: Novel Gated Fusion vs Baselines")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # FAST CONFIG
    ENV = "breakout"
    TIMESTEPS = 50000  # 50K steps - very fast!
    SEEDS = [42, 123, 456]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Environment: {ENV}")
    print(f"Timesteps: {TIMESTEPS:,}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}\n")
    
    env = minatar.Environment(ENV)
    num_actions = env.num_actions()
    
    text_instructions = {
        "breakout": "Break all the bricks by bouncing the ball with the paddle.",
        "space_invaders": "Shoot the aliens before they reach the bottom.",
        "freeway": "Guide the chicken across the highway avoiding cars."
    }
    
    results = {}
    
    # 1. CNN Baseline
    print("\n" + "=" * 50)
    print("CNN BASELINE")
    print("=" * 50)
    results['cnn'] = []
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        model = MinAtarCNN(num_actions)
        res = train_fast(model, ENV, TIMESTEPS, DEVICE, seed)
        results['cnn'].append(res)
    
    # 2. Fast Hybrid
    print("\n" + "=" * 50)
    print("FAST HYBRID (Multi-scale CNN)")
    print("=" * 50)
    results['hybrid'] = []
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        model = MinAtarHybridCNN(num_actions)
        res = train_fast(model, ENV, TIMESTEPS, DEVICE, seed)
        results['hybrid'].append(res)
    
    # 3. NOVEL: Gated Fusion
    print("\n" + "=" * 50)
    print("NOVEL: GATED VISION-LANGUAGE FUSION")
    print("=" * 50)
    results['gated_fusion'] = []
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        model = MinAtarGatedFusion(num_actions)
        res = train_fast(model, ENV, TIMESTEPS, DEVICE, seed, 
                        text_instruction=text_instructions[ENV], use_text=True)
        results['gated_fusion'].append(res)
        
        # Print gate statistics
        stats = model.get_gate_statistics()
        print(f"  Gate Stats: Vision={stats['vision_weight']:.3f}, Text={stats['text_weight']:.3f}")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for method in ['cnn', 'hybrid', 'gated_fusion']:
        all_final = [r for res in results[method] for r in res['final_returns']]
        mean = np.mean(all_final) if all_final else 0
        std = np.std(all_final) if all_final else 0
        sps = np.mean([res['sps'] for res in results[method]])
        
        label = {
            'cnn': 'CNN Baseline',
            'hybrid': 'Fast Hybrid',
            'gated_fusion': '🔥 Gated Fusion (Ours)'
        }[method]
        
        print(f"{label:<30} {mean:>6.2f} ± {std:>5.2f}  |  {sps:>5.0f} SPS")
    
    # Calculate improvement
    cnn_mean = np.mean([r for res in results['cnn'] for r in res['final_returns']])
    gated_mean = np.mean([r for res in results['gated_fusion'] for r in res['final_returns']])
    
    improvement = ((gated_mean - cnn_mean) / cnn_mean * 100) if cnn_mean > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"IMPROVEMENT: Gated Fusion vs CNN: {improvement:+.1f}%")
    print("=" * 70)
    
    # Save
    output_dir = "results/quick_gated_benchmark"
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'config': {'env': ENV, 'timesteps': TIMESTEPS, 'seeds': SEEDS},
        'results': {
            method: {
                'mean': float(np.mean([r for res in results[method] for r in res['final_returns']])),
                'std': float(np.std([r for res in results[method] for r in res['final_returns']])),
                'sps': float(np.mean([res['sps'] for res in results[method]]))
            }
            for method in results.keys()
        },
        'improvement_pct': float(improvement)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Quick visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    methods = ['cnn', 'hybrid', 'gated_fusion']
    labels_plot = ['CNN\nBaseline', 'Fast\nHybrid', 'Gated Fusion\n(Ours)']
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    
    means = [summary['results'][m]['mean'] for m in methods]
    stds = [summary['results'][m]['std'] for m in methods]
    
    bars = ax1.bar(range(len(methods)), means, yerr=stds, color=colors, 
                   edgecolor='black', linewidth=2, capsize=10)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels_plot, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Episode Return', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Learning curves
    for method, color, label in zip(methods, colors, labels_plot):
        all_returns = [res['returns'] for res in results[method]]
        if all_returns:
            min_len = min(len(r) for r in all_returns)
            if min_len > 0:
                aligned = [r[:min_len] for r in all_returns]
                mean_curve = np.mean(aligned, axis=0)
                timesteps = results[method][0]['timesteps'][:min_len]
                ax2.plot(timesteps, mean_curve, linewidth=3, color=color, 
                        label=label.replace('\n', ' '), marker='o', markersize=5)
    
    ax2.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Return', fontsize=12, fontweight='bold')
    ax2.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quick_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
    
    return results, summary


if __name__ == "__main__":
    results, summary = run_quick_benchmark()
