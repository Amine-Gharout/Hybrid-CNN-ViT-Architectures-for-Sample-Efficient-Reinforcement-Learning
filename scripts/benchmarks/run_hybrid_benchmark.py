"""
Full Benchmark: CNN vs ViT vs Hybrid vs ViT+Text
Generates all visualizations and comparison metrics
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# Environment
try:
    import minatar
    MINATAR_AVAILABLE = True
except ImportError:
    MINATAR_AVAILABLE = False
    print("MinAtar not available")

# Models
from models.cnn_policy import CNNPolicy, SmallCNNPolicy
from models.vit_policy import ActorCriticViT
from models.hybrid_policy import HybridCNNViT, HybridCNNViTSmall
from models.vit_text_policy import ActorCriticViTText

# Ensure output encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'


def preprocess_minatar_to_224(obs):
    """Convert MinAtar (10,10,C) -> (3, 224, 224)"""
    if obs.ndim == 2:
        obs = np.stack([obs]*3, axis=-1)
    elif obs.ndim == 3 and obs.shape[-1] > 3:
        obs = obs[:, :, :3]
    elif obs.ndim == 3 and obs.shape[-1] < 3:
        obs = np.concatenate([obs] * (3 // obs.shape[-1] + 1), axis=-1)[:, :, :3]
    
    obs_resized = cv2.resize(obs.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
    obs_norm = obs_resized / (obs_resized.max() + 1e-8)
    obs_chw = obs_norm.transpose(2, 0, 1)
    return obs_chw.astype(np.float32)


def train_with_logging(
    model,
    env_name,
    total_timesteps,
    learning_rate=2.5e-4,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    log_interval=10,
    device="cpu",
    seed=42,
    text_instruction=None,
):
    """Train with detailed logging for visualization"""
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = minatar.Environment(env_name)
    
    # Model setup
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    
    # Training metrics
    all_returns = []
    timesteps_log = []
    episode_returns = []
    current_episode_reward = 0
    
    # Rollout storage
    batch_size = num_steps
    minibatch_size = batch_size // num_minibatches
    
    # MinAtar reset returns None, state is accessed via env.state()
    env.reset()
    obs = env.state()
    obs_processed = preprocess_minatar_to_224(obs)
    obs_tensor = torch.tensor(obs_processed, dtype=torch.float32).unsqueeze(0).to(device)
    
    global_step = 0
    start_time = time.time()
    num_updates = total_timesteps // batch_size
    
    for update in range(1, num_updates + 1):
        # Storage
        obs_batch = []
        actions_batch = []
        log_probs_batch = []
        rewards_batch = []
        dones_batch = []
        values_batch = []
        
        # Collect rollout
        for step in range(num_steps):
            global_step += 1
            
            with torch.no_grad():
                if text_instruction is not None:
                    action, log_prob, _, value = model.get_action_and_value(
                        obs_tensor, [text_instruction]
                    )
                else:
                    action, log_prob, _, value = model.get_action_and_value(obs_tensor)
            
            obs_batch.append(obs_tensor.squeeze(0))
            actions_batch.append(action)
            log_probs_batch.append(log_prob)
            values_batch.append(value.flatten())
            
            # Step environment
            action_int = action.item()
            reward, terminated = env.act(action_int)
            current_episode_reward += reward
            
            rewards_batch.append(torch.tensor([reward], dtype=torch.float32, device=device))
            dones_batch.append(torch.tensor([float(terminated)], dtype=torch.float32, device=device))
            
            if terminated:
                episode_returns.append(current_episode_reward)
                current_episode_reward = 0
                env.reset()
            
            obs = env.state()
            obs_processed = preprocess_minatar_to_224(obs)
            obs_tensor = torch.tensor(obs_processed, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Stack tensors
        obs_batch = torch.stack(obs_batch)
        actions_batch = torch.cat(actions_batch)
        log_probs_batch = torch.cat(log_probs_batch)
        rewards_batch = torch.cat(rewards_batch)
        dones_batch = torch.cat(dones_batch)
        values_batch = torch.cat(values_batch)
        
        # Compute returns and advantages (GAE)
        with torch.no_grad():
            if text_instruction is not None:
                next_value = model.get_value(obs_tensor, [text_instruction]).flatten()
            else:
                next_value = model.get_value(obs_tensor).flatten()
            
            advantages = torch.zeros_like(rewards_batch, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones_batch[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_batch[t + 1]
                    nextvalues = values_batch[t + 1]
                delta = rewards_batch[t] + gamma * nextvalues * nextnonterminal - values_batch[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values_batch
        
        # PPO Update
        b_obs = obs_batch
        b_actions = actions_batch
        b_log_probs = log_probs_batch
        b_advantages = advantages
        b_returns = returns
        b_values = values_batch
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Mini-batch updates
        batch_indices = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = batch_indices[start:end]
                
                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_log_probs = b_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                
                if text_instruction is not None:
                    texts = [text_instruction] * len(mb_indices)
                    _, new_log_prob, entropy, new_value = model.get_action_and_value(
                        mb_obs, texts, mb_actions
                    )
                else:
                    _, new_log_prob, entropy, new_value = model.get_action_and_value(
                        mb_obs, mb_actions
                    )
                
                # Policy loss
                log_ratio = new_log_prob - mb_log_probs
                ratio = log_ratio.exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                new_value = new_value.flatten()
                v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        
        # Log progress
        if update % log_interval == 0 or update == 1:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            
            if len(episode_returns) > 0:
                mean_return = np.mean(episode_returns[-20:])
                all_returns.append(mean_return)
                timesteps_log.append(global_step)
                print(f"  Step {global_step:6d} | Return: {mean_return:.2f} | SPS: {sps}")
    
    return {
        'timesteps': timesteps_log,
        'returns': all_returns,
        'final_returns': episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns,
        'sps': sps
    }


def create_all_visualizations(results, output_dir):
    """Generate comprehensive visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color scheme
    colors = {
        'cnn': '#2ecc71',       # Green
        'vit': '#3498db',       # Blue  
        'hybrid': '#e74c3c',    # Red (main model!)
        'vit_text': '#9b59b6',  # Purple
    }
    
    labels = {
        'cnn': 'CNN (Baseline)',
        'vit': 'ViT',
        'hybrid': 'Hybrid CNN-ViT',
        'vit_text': 'ViT + Text'
    }
    
    # 1. Learning Curves
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in results:
        for seed_result in results[method]:
            timesteps = seed_result['timesteps']
            returns = seed_result['returns']
            if len(timesteps) > 0:
                ax.plot(timesteps, returns, alpha=0.3, color=colors.get(method, 'gray'))
        
        # Average
        all_returns = [r['returns'] for r in results[method] if len(r['returns']) > 0]
        if all_returns:
            min_len = min(len(r) for r in all_returns)
            if min_len > 0:
                aligned = [r[:min_len] for r in all_returns]
                mean_returns = np.mean(aligned, axis=0)
                timesteps = results[method][0]['timesteps'][:min_len]
                ax.plot(timesteps, mean_returns, linewidth=3, 
                       color=colors.get(method, 'gray'), 
                       label=labels.get(method, method))
    
    ax.set_xlabel('Timesteps', fontsize=14)
    ax.set_ylabel('Mean Episode Return', fontsize=14)
    ax.set_title('Learning Curves: CNN vs ViT vs Hybrid', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150)
    plt.close()
    print(f"  Saved: learning_curves.png")
    
    # 2. Final Performance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    means = []
    stds = []
    bar_colors = []
    
    for method in methods:
        final_returns = []
        for seed_result in results[method]:
            if len(seed_result['final_returns']) > 0:
                final_returns.extend(seed_result['final_returns'])
        
        if final_returns:
            means.append(np.mean(final_returns))
            stds.append(np.std(final_returns))
        else:
            means.append(0)
            stds.append(0)
        bar_colors.append(colors.get(method, 'gray'))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in methods], fontsize=12)
    ax.set_ylabel('Mean Episode Return', fontsize=14)
    ax.set_title('Final Performance Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.05,
               f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_performance.png'), dpi=150)
    plt.close()
    print(f"  Saved: final_performance.png")
    
    # 3. Return Distribution (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_for_box = []
    labels_for_box = []
    
    for method in methods:
        all_final = []
        for seed_result in results[method]:
            all_final.extend(seed_result['final_returns'])
        if all_final:
            data_for_box.append(all_final)
            labels_for_box.append(labels.get(method, method))
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(colors.get(method, 'gray'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Episode Return', fontsize=14)
    ax.set_title('Return Distribution', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'return_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: return_distribution.png")
    
    # 4. Speed Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sps_values = []
    for method in methods:
        method_sps = [r['sps'] for r in results[method]]
        sps_values.append(np.mean(method_sps))
    
    bars = ax.bar(x, sps_values, color=bar_colors, edgecolor='black', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in methods], fontsize=12)
    ax.set_ylabel('Steps Per Second (SPS)', fontsize=14)
    ax.set_title('Training Speed Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, sps in zip(bars, sps_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{sps:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: speed_comparison.png")
    
    # 5. Dashboard
    fig = plt.figure(figsize=(16, 12))
    
    # Learning curves
    ax1 = fig.add_subplot(2, 2, 1)
    for method in results:
        all_returns = [r['returns'] for r in results[method] if len(r['returns']) > 0]
        if all_returns:
            min_len = min(len(r) for r in all_returns)
            if min_len > 0:
                aligned = [r[:min_len] for r in all_returns]
                mean_returns = np.mean(aligned, axis=0)
                timesteps = results[method][0]['timesteps'][:min_len]
                ax1.plot(timesteps, mean_returns, linewidth=2, 
                        color=colors.get(method, 'gray'), label=labels.get(method, method))
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mean Return')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels.get(m, m) for m in methods], fontsize=10, rotation=15)
    ax2.set_ylabel('Mean Return')
    ax2.set_title('Final Performance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax3 = fig.add_subplot(2, 2, 3)
    bp = ax3.boxplot(data_for_box, labels=[labels.get(m, m) for m in methods[:len(data_for_box)]], patch_artist=True)
    for patch, method in zip(bp['boxes'], methods[:len(data_for_box)]):
        patch.set_facecolor(colors.get(method, 'gray'))
        patch.set_alpha(0.7)
    ax3.set_ylabel('Episode Return')
    ax3.set_title('Return Distribution')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Speed
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(x, sps_values, color=bar_colors, edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels([labels.get(m, m) for m in methods], fontsize=10, rotation=15)
    ax4.set_ylabel('Steps Per Second')
    ax4.set_title('Training Speed')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('ViT vs CNN vs Hybrid Benchmark Results', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: benchmark_dashboard.png")
    
    return means, stds, sps_values


class SmallViT(nn.Module):
    """Smaller/Faster ViT using tiny model with more frozen layers"""
    
    def __init__(self, num_actions: int):
        super().__init__()
        import timm
        
        # Use tiny ViT - much faster
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0,
        )
        
        # Freeze ALL ViT layers for speed - only train head
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Get output dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            vit_dim = self.vit(dummy).shape[-1]
        
        # Trainable head
        self.policy_net = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        print(f"SmallViT created - ViT frozen, only head trainable")
    
    def forward(self, obs):
        with torch.no_grad():
            features = self.vit(obs)
        logits = self.policy_net(features)
        value = self.value_net(features)
        return logits, value
    
    def get_value(self, obs):
        with torch.no_grad():
            features = self.vit(obs)
        return self.value_net(features)
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def run_full_benchmark():
    """Run the complete benchmark - Publication Ready"""
    
    print("=" * 60)
    print("FULL BENCHMARK: CNN vs ViT vs Hybrid vs ViT+Text")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ===== FAST PUBLICATION-READY CONFIGURATION =====
    ENVIRONMENTS = ["breakout", "space_invaders", "freeway"]  # Multiple games
    TOTAL_TIMESTEPS = 200000  # 200K steps - good for MinAtar
    SEEDS = [42, 123, 456]  # 3 seeds for statistical validity
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Environments: {ENVIRONMENTS}")
    print(f"Timesteps per run: {TOTAL_TIMESTEPS:,}")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} seeds)")
    print(f"Device: {DEVICE}")
    print()
    
    # Store all results
    all_env_results = {}
    
    for ENV_NAME in ENVIRONMENTS:
        print("\n" + "#" * 60)
        print(f"# ENVIRONMENT: {ENV_NAME.upper()}")
        print("#" * 60)
        
        # Get action space
        env = minatar.Environment(ENV_NAME)
        num_actions = env.num_actions()
        print(f"Actions: {num_actions}")
        
        results = {
            'cnn': [],
            'vit': [],
            'hybrid': [],
            'vit_text': []
        }
        
        # 1. CNN Baseline
        print("\n" + "=" * 50)
        print("Training CNN (Baseline)")
        print("=" * 50)
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            model = SmallCNNPolicy(num_actions)
            result = train_with_logging(
                model, ENV_NAME, TOTAL_TIMESTEPS,
                device=DEVICE, seed=seed
            )
            results['cnn'].append(result)
        
        # 2. ViT (Using smaller/faster version)
        print("\n" + "=" * 50)
        print("Training ViT (Tiny, Frozen Backbone)")
        print("=" * 50)
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            model = SmallViT(num_actions)
            result = train_with_logging(
                model, ENV_NAME, TOTAL_TIMESTEPS,
                device=DEVICE, seed=seed
            )
            results['vit'].append(result)
        
        # 3. Hybrid CNN-ViT (MAIN MODEL)
        print("\n" + "=" * 50)
        print("Training HYBRID CNN-ViT (Our Method)")
        print("=" * 50)
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            model = HybridCNNViTSmall(num_actions)
            result = train_with_logging(
                model, ENV_NAME, TOTAL_TIMESTEPS,
                device=DEVICE, seed=seed
            )
            results['hybrid'].append(result)
        
        # 4. ViT + Text
        print("\n" + "=" * 50)
        print("Training ViT + Text")
        print("=" * 50)
        
        text_instructions = {
            "breakout": "Break all the bricks by bouncing the ball with the paddle.",
            "space_invaders": "Shoot the aliens before they reach the bottom. Avoid their bullets.",
            "freeway": "Guide the chicken across the highway while avoiding cars."
        }
        
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            model = ActorCriticViTText(num_actions)
            result = train_with_logging(
                model, ENV_NAME, TOTAL_TIMESTEPS,
                device=DEVICE, seed=seed,
                text_instruction=text_instructions.get(ENV_NAME, "Play the game optimally.")
            )
            results['vit_text'].append(result)
        
        all_env_results[ENV_NAME] = results
    
    # Generate visualizations for each environment
    print("\n" + "=" * 50)
    print("Generating Visualizations")
    print("=" * 50)
    
    output_dir = "results/full_benchmark"
    os.makedirs(output_dir, exist_ok=True)
    
    all_summaries = {}
    
    for ENV_NAME in ENVIRONMENTS:
        env_output_dir = os.path.join(output_dir, ENV_NAME)
        results = all_env_results[ENV_NAME]
        means, stds, sps_values = create_all_visualizations(results, env_output_dir)
        
        methods = list(results.keys())
        all_summaries[ENV_NAME] = {
            method: {
                'mean_return': float(mean),
                'std_return': float(std),
                'sps': float(sps)
            }
            for method, mean, std, sps in zip(methods, means, stds, sps_values)
        }
    
    # Create cross-environment comparison
    create_cross_env_visualization(all_env_results, ENVIRONMENTS, output_dir)
    
    # Save results
    print("\n" + "=" * 50)
    print("FINAL RESULTS - ALL ENVIRONMENTS")
    print("=" * 50)
    
    method_labels = {
        'cnn': 'CNN (Baseline)',
        'vit': 'ViT',
        'hybrid': 'Hybrid CNN-ViT',
        'vit_text': 'ViT + Text'
    }
    
    for ENV_NAME in ENVIRONMENTS:
        print(f"\n--- {ENV_NAME.upper()} ---")
        results = all_env_results[ENV_NAME]
        methods = list(results.keys())
        
        print(f"{'Method':<20} {'Return':>12} {'SPS':>10}")
        print("-" * 45)
        
        for method in methods:
            final_returns = []
            for seed_result in results[method]:
                if len(seed_result['final_returns']) > 0:
                    final_returns.extend(seed_result['final_returns'])
            
            mean = np.mean(final_returns) if final_returns else 0
            std = np.std(final_returns) if final_returns else 0
            sps = np.mean([r['sps'] for r in results[method]])
            
            print(f"{method_labels[method]:<20} {mean:>8.2f} +/- {std:<5.2f} {sps:>8.0f}")
    
    # Calculate overall improvements
    print("\n" + "=" * 50)
    print("KEY FINDINGS (Averaged Across Environments)")
    print("=" * 50)
    
    overall_hybrid_vs_cnn = []
    overall_hybrid_vs_vit = []
    
    for ENV_NAME in ENVIRONMENTS:
        results = all_env_results[ENV_NAME]
        
        cnn_returns = [r for s in results['cnn'] for r in s['final_returns']]
        vit_returns = [r for s in results['vit'] for r in s['final_returns']]
        hybrid_returns = [r for s in results['hybrid'] for r in s['final_returns']]
        
        cnn_mean = np.mean(cnn_returns) if cnn_returns else 0
        vit_mean = np.mean(vit_returns) if vit_returns else 0
        hybrid_mean = np.mean(hybrid_returns) if hybrid_returns else 0
        
        if cnn_mean > 0:
            overall_hybrid_vs_cnn.append((hybrid_mean - cnn_mean) / cnn_mean * 100)
        if vit_mean > 0:
            overall_hybrid_vs_vit.append((hybrid_mean - vit_mean) / vit_mean * 100)
    
    avg_vs_cnn = np.mean(overall_hybrid_vs_cnn) if overall_hybrid_vs_cnn else 0
    avg_vs_vit = np.mean(overall_hybrid_vs_vit) if overall_hybrid_vs_vit else 0
    
    print(f"Hybrid vs CNN (avg): {avg_vs_cnn:+.1f}%")
    print(f"Hybrid vs ViT (avg): {avg_vs_vit:+.1f}%")
    
    # Save comprehensive summary
    summary = {
        'config': {
            'environments': ENVIRONMENTS,
            'timesteps': TOTAL_TIMESTEPS,
            'seeds': SEEDS,
            'num_seeds': len(SEEDS)
        },
        'per_environment': all_summaries,
        'overall_improvements': {
            'hybrid_vs_cnn_avg': float(avg_vs_cnn),
            'hybrid_vs_vit_avg': float(avg_vs_vit)
        }
    }
    
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_env_results, summary


def create_cross_env_visualization(all_env_results, environments, output_dir):
    """Create visualization comparing performance across environments"""
    
    colors = {
        'cnn': '#2ecc71',
        'vit': '#3498db',
        'hybrid': '#e74c3c',
        'vit_text': '#9b59b6',
    }
    
    labels = {
        'cnn': 'CNN',
        'vit': 'ViT',
        'hybrid': 'Hybrid',
        'vit_text': 'ViT+Text'
    }
    
    methods = ['cnn', 'vit', 'hybrid', 'vit_text']
    
    fig, axes = plt.subplots(1, len(environments), figsize=(5*len(environments), 6))
    if len(environments) == 1:
        axes = [axes]
    
    for ax, env_name in zip(axes, environments):
        results = all_env_results[env_name]
        
        means = []
        stds = []
        
        for method in methods:
            final_returns = [r for s in results[method] for r in s['final_returns']]
            means.append(np.mean(final_returns) if final_returns else 0)
            stds.append(np.std(final_returns) if final_returns else 0)
        
        x = np.arange(len(methods))
        bar_colors = [colors[m] for m in methods]
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in methods], fontsize=10)
        ax.set_ylabel('Mean Return')
        ax.set_title(f'{env_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Performance Across MinAtar Environments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_environment_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: cross_environment_comparison.png")


if __name__ == "__main__":
    results, summary = run_full_benchmark()
