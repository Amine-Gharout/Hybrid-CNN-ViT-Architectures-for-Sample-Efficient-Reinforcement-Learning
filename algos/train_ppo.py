"""
PPO Training Script for ViT and Multimodal Policies
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from envs.atari_env import make_parallel_envs
from models.vit_policy import ActorCriticViT
from models.mlp_policy import MLPPolicy
from models.vit_text_policy import ActorCriticViTText
from utils.text_history import ParallelHistoryBuffer
from evaluation.metrics import MetricsCollector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--condition", type=str, default="vit_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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


def create_policy(args, config: Dict, num_actions: int, device: str, obs_space):
    """Create policy based on condition and environment"""
    condition_config = config['conditions'][args.condition]
    model_config = config['model']
    
    use_text = condition_config.get('use_text', False)
    
    # Check if visual environment
    is_visual = len(obs_space.shape) >= 3  # H x W x C
    
    if not is_visual:
        # Use simple MLP for non-visual environments
        print(f"Using MLP policy for non-visual environment")
        # Get observation dimension
        if hasattr(obs_space, 'shape') and len(obs_space.shape) > 0:
            obs_dim = obs_space.shape[-1] if len(obs_space.shape) > 1 else obs_space.shape[0]
        else:
            obs_dim = obs_space.n if hasattr(obs_space, 'n') else 4
        
        policy = MLPPolicy(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=64,
        ).to(device)
        use_text = False  # No text for simple envs
    elif use_text:
        policy = ActorCriticViTText(
            num_actions=num_actions,
            vit_model=model_config['vit_model'],
            text_model=model_config['text_model'],
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            fusion_type=condition_config.get('fusion_type', 'concat'),
            pretrained=condition_config.get('pretrained', True),
            freeze_vit_layers=model_config['freeze_vit_layers'],
            freeze_text=model_config['freeze_text'],
        ).to(device)
    else:
        policy = ActorCriticViT(
            num_actions=num_actions,
            vit_model=model_config['vit_model'],
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            pretrained=condition_config.get('pretrained', True),
            freeze_vit_layers=model_config['freeze_vit_layers'],
        ).to(device)
    
    return policy, use_text


def train(args):
    # Load config
    config = load_config(args.config)
    train_config = config['training']
    condition_config = config['conditions'][args.condition]
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if specified
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={**config, 'env': args.env, 'seed': args.seed, 'condition': args.condition}
        )
    
    # TensorBoard
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    
    # Create environments
    envs = make_parallel_envs(
        env_name=args.env,
        num_envs=train_config['num_envs'],
        seed=args.seed,
        vit_mode=True,
    )
    
    # Get action space size
    if hasattr(envs.action_space, 'n'):
        num_actions = envs.action_space.n
    elif hasattr(envs.action_space, 'nvec'):
        num_actions = int(envs.action_space.nvec[0])
    else:
        raise ValueError(f"Unsupported action space: {envs.action_space}")
    
    # Get observation space (use single_observation_space for vectorized envs)
    obs_space = envs.single_observation_space if hasattr(envs, 'single_observation_space') else envs.observation_space
    
    # Check if visual environment  
    is_visual = len(obs_space.shape) >= 3  # H x W x C
    
    print(f"Environment: {args.env}")
    print(f"  Observation space: {obs_space}")
    print(f"  Action space: {envs.action_space} (n={num_actions})")
    print(f"  Visual environment: {is_visual}")
    
    # Create policy
    policy, use_text = create_policy(args, config, num_actions, args.device, obs_space)
    
    # Create history buffer if using text
    history_buffer = None
    if use_text:
        history_buffer = ParallelHistoryBuffer(
            num_envs=train_config['num_envs'],
            history_length=condition_config.get('history_length', 5),
            template_style=condition_config.get('text_style', 'detailed'),
        )
    
    # Optimizer
    optimizer = optim.Adam(
        policy.parameters(),
        lr=train_config['learning_rate'],
        eps=train_config['adam_epsilon'],
    )
    
    # Metrics
    metrics_collector = MetricsCollector()
    
    # Training storage
    num_steps = train_config['num_steps']
    num_envs = train_config['num_envs']
    
    # Determine observation shape
    if is_visual:
        obs_shape = (3, 224, 224)
    else:
        # For vector observations
        obs_shape = obs_space.shape
    
    obs_storage = torch.zeros((num_steps, num_envs, *obs_shape)).to(args.device)
    actions_storage = torch.zeros((num_steps, num_envs)).to(args.device)
    logprobs_storage = torch.zeros((num_steps, num_envs)).to(args.device)
    rewards_storage = torch.zeros((num_steps, num_envs)).to(args.device)
    dones_storage = torch.zeros((num_steps, num_envs)).to(args.device)
    values_storage = torch.zeros((num_steps, num_envs)).to(args.device)
    
    if use_text:
        text_storage = [[None for _ in range(num_envs)] for _ in range(num_steps)]
    
    # Initialize environment
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(num_envs).to(args.device)
    
    num_updates = train_config['total_timesteps'] // (num_steps * num_envs)
    
    print(f"Training {args.condition} on {args.env} for {num_updates} updates")
    print(f"Device: {args.device}")
    print(f"Use text: {use_text}")
    
    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if train_config.get('lr_schedule') == 'linear':
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * train_config['learning_rate']
            optimizer.param_groups[0]['lr'] = lr_now
        
        # Collect rollout
        for step in range(num_steps):
            global_step += num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done
            
            # Get action
            with torch.no_grad():
                if use_text:
                    text_batch = history_buffer.to_text_batch()
                    text_storage[step] = text_batch
                    action, logprob, _, value = policy.get_action_and_value(next_obs, text_batch)
                else:
                    action, logprob, _, value = policy.get_action_and_value(next_obs)
                
                values_storage[step] = value.flatten()
            
            actions_storage[step] = action
            logprobs_storage[step] = logprob
            
            # Execute action
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards_storage[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs = torch.Tensor(next_obs).to(args.device)
            next_done = torch.Tensor(done).to(args.device)
            
            # Update history buffer
            if use_text:
                history_buffer.add(
                    actions=action.cpu().numpy(),
                    rewards=reward,
                    dones=done,
                    infos=infos if isinstance(infos, list) else [infos]
                )
            
            # Log episode info
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = info["episode"]["r"]
                        ep_length = info["episode"]["l"]
                        print(f"Step {global_step}: Episode return = {ep_return:.2f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        
                        if args.wandb_project:
                            wandb.log({
                                "episodic_return": ep_return,
                                "episodic_length": ep_length,
                                "global_step": global_step
                            })
        
        # Bootstrap value
        with torch.no_grad():
            if use_text:
                text_batch = history_buffer.to_text_batch()
                next_value = policy.get_value(next_obs, text_batch).reshape(1, -1)
            else:
                next_value = policy.get_value(next_obs).reshape(1, -1)
            
            # Compute advantages (GAE)
            advantages = torch.zeros_like(rewards_storage).to(args.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    nextvalues = values_storage[t + 1]
                delta = rewards_storage[t] + train_config['gamma'] * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = lastgaelam = delta + train_config['gamma'] * train_config['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values_storage
        
        # Flatten batch
        b_obs = obs_storage.reshape((-1, *obs_shape))
        b_actions = actions_storage.reshape(-1)
        b_logprobs = logprobs_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)
        
        if use_text:
            b_text = [text_storage[i][j] for i in range(num_steps) for j in range(num_envs)]
        
        # Optimize policy
        b_inds = np.arange(num_steps * num_envs)
        clipfracs = []
        
        for epoch in range(train_config['num_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps * num_envs, train_config['minibatch_size']):
                end = start + train_config['minibatch_size']
                mb_inds = b_inds[start:end]
                
                if use_text:
                    mb_text = [b_text[i] for i in mb_inds]
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        b_obs[mb_inds],
                        mb_text,
                        b_actions.long()[mb_inds]
                    )
                else:
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions.long()[mb_inds]
                    )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > train_config['clip_coef']).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_config['clip_coef'], 1 + train_config['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - train_config['ent_coef'] * entropy_loss + v_loss * train_config['vf_coef']
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), train_config['max_grad_norm'])
                optimizer.step()
        
        # Logging
        if update % train_config['log_interval'] == 0:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            print(f"\nUpdate {update}/{num_updates}")
            print(f"SPS: {int(global_step / (time.time() - start_time))}")
            print(f"Policy loss: {pg_loss.item():.4f}")
            print(f"Value loss: {v_loss.item():.4f}")
            print(f"Entropy: {entropy_loss.item():.4f}")
            
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
            if args.wandb_project:
                wandb.log({
                    "policy_loss": pg_loss.item(),
                    "value_loss": v_loss.item(),
                    "entropy": entropy_loss.item(),
                    "explained_variance": explained_var,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "global_step": global_step
                })
        
        # Save checkpoint
        if update % train_config['save_interval'] == 0:
            checkpoint_path = output_dir / f"checkpoint_{update}.pt"
            torch.save({
                'update': update,
                'global_step': global_step,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    envs.close()
    writer.close()
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
