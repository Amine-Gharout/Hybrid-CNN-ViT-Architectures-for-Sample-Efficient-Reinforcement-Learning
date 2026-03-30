"""
Compare ViT vs CNN vs Multimodal on MinAtar
Run all three approaches and save comparison results
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

from minatar import Environment

from models.vit_policy import ActorCriticViT
from models.vit_text_policy import ActorCriticViTText
from models.cnn_policy import CNNPolicy, SmallCNNPolicy
from utils.text_history import HistoryBuffer


class MinAtarWrapper:
    """Wraps MinAtar environment for ViT/CNN input (3, 224, 224)"""
    
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


def train_policy(
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
):
    """Generic training function for any policy"""
    
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    
    if use_text:
        history = HistoryBuffer(history_length=5, template_style='detailed')
    
    # Storage
    obs_storage = torch.zeros((num_steps, 3, 224, 224)).to(device)
    actions_storage = torch.zeros(num_steps).to(device)
    logprobs_storage = torch.zeros(num_steps).to(device)
    rewards_storage = torch.zeros(num_steps).to(device)
    dones_storage = torch.zeros(num_steps).to(device)
    values_storage = torch.zeros(num_steps).to(device)
    
    if use_text:
        text_storage = [None] * num_steps
    
    # Initialize
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs).to(device) / 255.0
    done = False
    
    episode_returns = []
    episode_lengths = []
    current_return = 0
    current_length = 0
    
    num_updates = total_timesteps // num_steps
    start_time = time.time()
    
    for update in range(1, num_updates + 1):
        for step in range(num_steps):
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
        
        # Optimize
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
        
        if update % 20 == 0:
            elapsed = time.time() - start_time
            sps = int((update * num_steps) / elapsed)
            if episode_returns:
                mean_ret = np.mean(episode_returns[-10:])
                print(f"  [{policy_name}] Update {update}/{num_updates} | SPS: {sps} | Return: {mean_ret:.2f}")
    
    training_time = time.time() - start_time
    
    return {
        'policy_name': policy_name,
        'episode_returns': episode_returns,
        'mean_return': float(np.mean(episode_returns)) if episode_returns else 0,
        'std_return': float(np.std(episode_returns)) if episode_returns else 0,
        'max_return': float(np.max(episode_returns)) if episode_returns else 0,
        'total_episodes': len(episode_returns),
        'training_time': training_time,
        'sps': int(total_timesteps / training_time),
    }


def run_comparison(game='breakout', seed=42, timesteps=50000):
    """Run comparison between CNN, ViT, and ViT+Text"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("ViT vs CNN vs Multimodal COMPARISON BENCHMARK")
    print("="*70)
    print(f"Game: {game}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Device: {device}")
    print("="*70)
    
    results = []
    
    # Create environment to get num_actions
    env = MinAtarWrapper(game=game, seed=seed)
    num_actions = env.action_space_n
    env.close()
    
    # ========================================
    # 1. CNN Baseline (Classical RL)
    # ========================================
    print("\n" + "-"*70)
    print("TRAINING: CNN Baseline (Classical RL)")
    print("-"*70)
    
    env = MinAtarWrapper(game=game, seed=seed)
    cnn_policy = SmallCNNPolicy(
        num_actions=num_actions,
        input_channels=3,
        hidden_dim=128,
    ).to(device)
    
    cnn_results = train_policy(
        policy=cnn_policy,
        env=env,
        policy_name="CNN",
        use_text=False,
        total_timesteps=timesteps,
        device=device,
    )
    cnn_results['parameters'] = sum(p.numel() for p in cnn_policy.parameters())
    results.append(cnn_results)
    env.close()
    
    print(f"\n  CNN Complete: Mean={cnn_results['mean_return']:.2f}, Time={cnn_results['training_time']:.0f}s")
    
    del cnn_policy
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # 2. ViT-Only
    # ========================================
    print("\n" + "-"*70)
    print("TRAINING: ViT-Only (Vision Transformer)")
    print("-"*70)
    
    env = MinAtarWrapper(game=game, seed=seed)
    vit_policy = ActorCriticViT(
        num_actions=num_actions,
        vit_model="vit_tiny_patch16_224",
        embedding_dim=192,
        hidden_dim=128,
        pretrained=True,
        freeze_vit_layers=8,
    ).to(device)
    
    vit_results = train_policy(
        policy=vit_policy,
        env=env,
        policy_name="ViT",
        use_text=False,
        total_timesteps=timesteps,
        device=device,
    )
    vit_results['parameters'] = sum(p.numel() for p in vit_policy.parameters())
    results.append(vit_results)
    env.close()
    
    print(f"\n  ViT Complete: Mean={vit_results['mean_return']:.2f}, Time={vit_results['training_time']:.0f}s")
    
    del vit_policy
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # 3. ViT + Text (Multimodal)
    # ========================================
    print("\n" + "-"*70)
    print("TRAINING: ViT + Text (Multimodal)")
    print("-"*70)
    
    env = MinAtarWrapper(game=game, seed=seed)
    mm_policy = ActorCriticViTText(
        num_actions=num_actions,
        vit_model="vit_tiny_patch16_224",
        text_model="distilbert-base-uncased",
        embedding_dim=192,
        hidden_dim=128,
        fusion_type="concat",
        pretrained=True,
        freeze_vit_layers=8,
        freeze_text=True,
    ).to(device)
    
    mm_results = train_policy(
        policy=mm_policy,
        env=env,
        policy_name="ViT+Text",
        use_text=True,
        total_timesteps=timesteps,
        device=device,
    )
    mm_results['parameters'] = sum(p.numel() for p in mm_policy.parameters())
    results.append(mm_results)
    env.close()
    
    print(f"\n  ViT+Text Complete: Mean={mm_results['mean_return']:.2f}, Time={mm_results['training_time']:.0f}s")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Method':<15} {'Params':>12} {'Mean Return':>12} {'Max':>8} {'Time':>10} {'SPS':>8}")
    print("-"*70)
    
    for r in results:
        print(f"{r['policy_name']:<15} {r['parameters']:>12,} {r['mean_return']:>12.2f} "
              f"{r['max_return']:>8.1f} {r['training_time']:>9.0f}s {r['sps']:>8}")
    
    print("-"*70)
    
    # Calculate relative performance
    cnn_mean = results[0]['mean_return']
    vit_mean = results[1]['mean_return']
    mm_mean = results[2]['mean_return']
    
    print(f"\nRelative to CNN baseline:")
    print(f"  ViT:      {((vit_mean/cnn_mean)-1)*100:+.1f}%")
    print(f"  ViT+Text: {((mm_mean/cnn_mean)-1)*100:+.1f}%")
    
    print("="*70)
    
    # Save results
    output = {
        'game': game,
        'seed': seed,
        'timesteps': timesteps,
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }
    
    filename = f"comparison_{game}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timesteps', type=int, default=50000)
    args = parser.parse_args()
    
    run_comparison(
        game=args.game,
        seed=args.seed,
        timesteps=args.timesteps,
    )
