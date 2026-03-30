"""
MinAtar Benchmark for ViT RL
MinAtar provides simplified Atari-like visual environments that work with Python 3.14
Games: Breakout, Freeway, Asterix, Seaquest, Space_invaders
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

# MinAtar environments
from minatar import Environment

# Our models
from models.vit_policy import ActorCriticViT
from models.vit_text_policy import ActorCriticViTText
from utils.text_history import HistoryBuffer


class MinAtarWrapper:
    """Wraps MinAtar environment for ViT input (3, 224, 224)"""
    
    def __init__(self, game='breakout', seed=42):
        self.env = Environment(game)
        self.env.seed(seed)
        self.game = game
        self.action_space_n = self.env.num_actions()
        
        # MinAtar state is (H, W, C) - typically (10, 10, n_channels)
        # We need to resize to (224, 224) for ViT
        
    def reset(self):
        self.env.reset()
        state = self.env.state()
        return self._process_state(state), {}
    
    def step(self, action):
        reward, done = self.env.act(action)
        state = self.env.state()
        return self._process_state(state), reward, done, False, {}
    
    def _process_state(self, state):
        """Convert MinAtar state to ViT-compatible format"""
        # state is (H, W, C) numpy array
        # Convert to RGB and resize to 224x224
        import cv2
        
        # Sum channels to create grayscale-ish, then convert to RGB
        if state.ndim == 3:
            # Create RGB from channels
            h, w, c = state.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            
            # Map different channels to RGB colors
            for i in range(min(c, 3)):
                rgb[:, :, i] = state[:, :, i].astype(np.float32)
            
            # Normalize and convert to uint8
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = (state * 255).clip(0, 255).astype(np.uint8)
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        
        # Resize to 224x224
        rgb_resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        # Convert to (C, H, W) format
        return rgb_resized.transpose(2, 0, 1)
    
    def close(self):
        pass


def train_minatar(
    game: str = 'breakout',
    condition: str = 'vit_only',
    seed: int = 42,
    total_timesteps: int = 100000,
    num_steps: int = 128,
    num_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    device: str = None,
):
    """Train ViT policy on MinAtar"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print(f"MinAtar ViT Benchmark")
    print("="*60)
    print(f"Game: {game}")
    print(f"Condition: {condition}")
    print(f"Seed: {seed}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print("="*60)
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = MinAtarWrapper(game=game, seed=seed)
    num_actions = env.action_space_n
    print(f"Actions: {num_actions}")
    
    # Create policy
    use_text = 'text' in condition
    
    if use_text:
        fusion_type = 'concat'
        if 'crossattn' in condition:
            fusion_type = 'cross_attention'
        elif 'gated' in condition:
            fusion_type = 'gated'
            
        policy = ActorCriticViTText(
            num_actions=num_actions,
            vit_model="vit_tiny_patch16_224",
            text_model="distilbert-base-uncased",
            embedding_dim=192,
            hidden_dim=128,
            fusion_type=fusion_type,
            pretrained=True,
            freeze_vit_layers=8,
            freeze_text=True,
        ).to(device)
    else:
        policy = ActorCriticViT(
            num_actions=num_actions,
            vit_model="vit_tiny_patch16_224",
            embedding_dim=192,
            hidden_dim=128,
            pretrained=True,
            freeze_vit_layers=8,
        ).to(device)
    
    print(f"Policy: {type(policy).__name__}")
    print(f"Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    
    # History buffer for text
    if use_text:
        history = HistoryBuffer(history_length=5, template_style='detailed')
    
    # Training storage
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
    
    # Tracking
    episode_returns = []
    episode_lengths = []
    current_return = 0
    current_length = 0
    
    num_updates = total_timesteps // num_steps
    start_time = time.time()
    
    print(f"\nStarting training for {num_updates} updates...")
    print("-"*60)
    
    for update in range(1, num_updates + 1):
        # Collect rollout
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
            
            # Step environment
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
        
        # Bootstrap value
        with torch.no_grad():
            if use_text:
                _, _, _, next_value = policy.get_action_and_value(obs.unsqueeze(0), [history.to_text()])
            else:
                _, _, _, next_value = policy.get_action_and_value(obs.unsqueeze(0))
            next_value = next_value.flatten()
        
        # Compute advantages (GAE)
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
        
        # Flatten
        b_obs = obs_storage
        b_actions = actions_storage
        b_logprobs = logprobs_storage
        b_advantages = advantages
        b_returns = returns
        b_values = values_storage
        
        if use_text:
            b_texts = text_storage
        
        # Optimize
        b_inds = np.arange(num_steps)
        
        for epoch in range(num_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]
                
                if use_text:
                    mb_texts = [b_texts[i] for i in mb_inds]
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        b_obs[mb_inds], mb_texts, b_actions[mb_inds].long()
                    )
                else:
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        b_obs[mb_inds], action=b_actions[mb_inds].long()
                    )
                
                # PPO loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        
        # Logging
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = int((update * num_steps) / elapsed)
            
            if episode_returns:
                mean_return = np.mean(episode_returns[-10:])
                print(f"Update {update}/{num_updates} | "
                      f"SPS: {sps} | "
                      f"Mean Return (last 10): {mean_return:.2f} | "
                      f"Episodes: {len(episode_returns)}")
            else:
                print(f"Update {update}/{num_updates} | SPS: {sps}")
    
    # Final results
    env.close()
    
    results = {
        'game': game,
        'condition': condition,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'mean_return': float(np.mean(episode_returns)) if episode_returns else 0,
        'std_return': float(np.std(episode_returns)) if episode_returns else 0,
        'max_return': float(np.max(episode_returns)) if episode_returns else 0,
        'total_episodes': len(episode_returns),
        'training_time': time.time() - start_time,
    }
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total episodes: {len(episode_returns)}")
    print(f"Mean return: {results['mean_return']:.2f} +/- {results['std_return']:.2f}")
    print(f"Max return: {results['max_return']:.2f}")
    print(f"Training time: {results['training_time']:.1f}s")
    print("="*60)
    
    return results


def run_benchmark(games=['breakout'], conditions=['vit_only', 'vit_text_concat'], 
                  seeds=[42, 123, 456], timesteps=100000):
    """Run full benchmark across games, conditions, and seeds"""
    
    all_results = []
    
    for game in games:
        for condition in conditions:
            for seed in seeds:
                print(f"\n{'#'*60}")
                print(f"# Running: {game} | {condition} | seed={seed}")
                print(f"{'#'*60}\n")
                
                results = train_minatar(
                    game=game,
                    condition=condition,
                    seed=seed,
                    total_timesteps=timesteps,
                )
                
                all_results.append(results)
    
    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nAll results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for game in games:
        print(f"\n{game.upper()}")
        for condition in conditions:
            game_cond_results = [r for r in all_results 
                                 if r['game'] == game and r['condition'] == condition]
            if game_cond_results:
                mean_returns = [r['mean_return'] for r in game_cond_results]
                print(f"  {condition}: {np.mean(mean_returns):.2f} +/- {np.std(mean_returns):.2f}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout',
                        choices=['breakout', 'freeway', 'asterix', 'seaquest', 'space_invaders'])
    parser.add_argument('--condition', type=str, default='vit_only',
                        choices=['vit_only', 'vit_text_concat', 'vit_text_crossattn', 'vit_text_gated'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--benchmark', action='store_true', help='Run full benchmark')
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(
            games=['breakout', 'freeway'],
            conditions=['vit_only', 'vit_text_concat'],
            seeds=[42, 123],
            timesteps=args.timesteps,
        )
    else:
        train_minatar(
            game=args.game,
            condition=args.condition,
            seed=args.seed,
            total_timesteps=args.timesteps,
        )
