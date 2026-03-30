"""
Evaluation script for trained policies
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from envs.atari_env import make_parallel_envs, AtariEnvWrapper
from models.vit_policy import ActorCriticViT
from models.mlp_policy import MLPPolicy
from models.vit_text_policy import ActorCriticViTText
from utils.text_history import HistoryBuffer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--condition", type=str, default="vit_only")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use-text", action="store_true")
    return parser.parse_args()


def evaluate_policy(
    policy,
    env_name: str,
    num_episodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
    use_text: bool = False,
    render: bool = False,
) -> Dict:
    """Evaluate a trained policy"""
    
    # Create single environment for evaluation
    import gymnasium as gym
    is_atari = "NoFrameskip" in env_name or "ALE" in env_name
    
    if is_atari:
        env = AtariEnvWrapper(
            env_name=env_name,
            seed=seed,
            vit_mode=True,
        )
    else:
        env = gym.make(env_name)
        env.reset(seed=seed)
    
    if use_text:
        history_buffer = HistoryBuffer(
            history_length=5,
            template_style="detailed",
        )
    
    episode_returns = []
    episode_lengths = []
    
    policy.eval()
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).unsqueeze(0).to(device)
        done = False
        episode_return = 0
        episode_length = 0
        
        if use_text:
            history_buffer.reset()
        
        while not done:
            with torch.no_grad():
                if use_text:
                    text = history_buffer.to_text()
                    action, _, _, _ = policy.get_action_and_value(obs, [text])
                else:
                    action, _, _, _ = policy.get_action_and_value(obs)
                
                action = action.cpu().numpy()[0]
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if use_text:
                history_buffer.add(
                    action=int(action),
                    reward=float(reward),
                    done=done,
                    info=info
                )
            
            episode_return += reward
            episode_length += 1
            
            obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Return = {episode_return:.2f}, Length = {episode_length}")
    
    env.close()
    
    results = {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_return': float(np.min(episode_returns)),
        'max_return': float(np.max(episode_returns)),
    }
    
    return results


def main():
    args = parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Determine number of actions and observation space from environment
    import gymnasium as gym
    is_atari = "NoFrameskip" in args.env or "ALE" in args.env
    
    if is_atari:
        temp_env = AtariEnvWrapper(env_name=args.env, seed=args.seed, vit_mode=True)
        num_actions = temp_env.action_space.n
        obs_space = temp_env.observation_space
        is_visual = True
        temp_env.close()
    else:
        temp_env = gym.make(args.env)
        if hasattr(temp_env.action_space, 'n'):
            num_actions = temp_env.action_space.n
        else:
            num_actions = 2
        obs_space = temp_env.observation_space
        is_visual = len(obs_space.shape) >= 3
        temp_env.close()
    
    # Create policy based on environment type
    if not is_visual:
        # Use MLP for non-visual environments
        obs_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape[-1]
        policy = MLPPolicy(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=64,
        ).to(args.device)
    elif args.use_text:
        policy = ActorCriticViTText(
            num_actions=num_actions,
            vit_model="vit_base_patch16_224",
            text_model="distilbert-base-uncased",
            embedding_dim=512,
            hidden_dim=256,
            fusion_type="concat",
        ).to(args.device)
    else:
        policy = ActorCriticViT(
            num_actions=num_actions,
            vit_model="vit_base_patch16_224",
            embedding_dim=512,
            hidden_dim=256,
        ).to(args.device)
    
    # Load weights
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    
    print(f"Evaluating on {args.env} for {args.num_episodes} episodes")
    
    # Evaluate
    results = evaluate_policy(
        policy=policy,
        env_name=args.env,
        num_episodes=args.num_episodes,
        seed=args.seed,
        device=args.device,
        use_text=args.use_text,
        render=args.render,
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Min Return: {results['min_return']:.2f}")
    print(f"Max Return: {results['max_return']:.2f}")
    print("="*60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
