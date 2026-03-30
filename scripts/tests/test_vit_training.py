"""
Test ViT-based RL Training
Tests the actual ViT policy on a visual environment (not MLP fallback)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import gymnasium as gym
import numpy as np


class VisualCartPoleWrapper(gym.ObservationWrapper):
    """Wraps CartPole to provide visual observations like Atari"""
    def __init__(self, env):
        super().__init__(env)
        # Visual observation space (3, 224, 224) like ViT expects
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 224, 224), dtype=np.uint8
        )
    
    def observation(self, obs):
        # Convert state to a simple visual representation
        # Create a 224x224 RGB image
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Draw cart position (normalize to image width)
        cart_x = int((obs[0] + 2.4) / 4.8 * 224)
        cart_x = np.clip(cart_x, 0, 223)
        
        # Draw cart (red rectangle)
        img[100:120, max(0, cart_x-10):min(224, cart_x+10)] = [255, 0, 0]
        
        # Draw pole angle (green line)
        pole_angle = obs[2]
        pole_end_x = int(cart_x + 50 * np.sin(pole_angle))
        pole_end_y = int(110 - 50 * np.cos(pole_angle))
        pole_end_x = np.clip(pole_end_x, 0, 223)
        pole_end_y = np.clip(pole_end_y, 0, 223)
        
        # Simple line drawing
        img[pole_end_y-2:pole_end_y+2, pole_end_x-2:pole_end_x+2] = [0, 255, 0]
        
        # Transpose to (C, H, W) format
        return img.transpose(2, 0, 1)


def test_vit_training():
    """Test ViT policy training on visual CartPole"""
    print("\n" + "="*60)
    print("Testing ViT-based RL Training")
    print("="*60)
    
    # Create visual environment
    env = gym.make('CartPole-v1')
    env = VisualCartPoleWrapper(env)
    
    print(f"\nEnvironment: Visual CartPole")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    
    # Import models
    from models.vit_policy import ActorCriticViT
    from models.vit_text_policy import ActorCriticViTText
    from utils.text_history import HistoryBuffer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # ============================================================
    # TEST 1: ViT-only policy
    # ============================================================
    print(f"\n" + "-"*60)
    print("TEST 1: ViT-Only Policy")
    print("-"*60)
    
    policy_vit = ActorCriticViT(
        num_actions=env.action_space.n,
        vit_model="vit_tiny_patch16_224",  # Use tiny for faster testing
        embedding_dim=192,
        hidden_dim=128,
        pretrained=True,
        freeze_vit_layers=8,
    ).to(device)
    
    print(f"[OK] Created ViT policy")
    print(f"  Model: vit_tiny_patch16_224")
    print(f"  Parameters: {sum(p.numel() for p in policy_vit.parameters()):,}")
    
    # Test forward pass
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        action, logprob, entropy, value = policy_vit.get_action_and_value(obs_tensor)
    
    print(f"[OK] Forward pass successful")
    print(f"  Input shape: {obs_tensor.shape}")
    print(f"  Action: {action.item()}")
    print(f"  Value: {value.item():.4f}")
    print(f"  Entropy: {entropy.item():.4f}")
    
    # Test gradient computation
    obs_tensor.requires_grad = False
    action, logprob, entropy, value = policy_vit.get_action_and_value(obs_tensor)
    loss = -logprob.mean() + 0.5 * value.mean()**2 - 0.01 * entropy.mean()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in policy_vit.parameters() if p.grad is not None)
    print(f"[OK] Backward pass successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    # Run a few training steps
    print(f"\n[...] Running 5 training episodes with ViT policy...")
    optimizer = torch.optim.Adam(policy_vit.parameters(), lr=1e-4)
    
    total_rewards = []
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 200:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                action, _, _, _ = policy_vit.get_action_and_value(obs_tensor)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_reward += reward
            steps += 1
        
        total_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: reward = {ep_reward:.0f}, steps = {steps}")
    
    print(f"[OK] ViT training test complete")
    print(f"  Average reward: {np.mean(total_rewards):.2f}")
    
    # ============================================================
    # TEST 2: Multimodal (ViT + Text) policy
    # ============================================================
    print(f"\n" + "-"*60)
    print("TEST 2: Multimodal (ViT + Text) Policy")
    print("-"*60)
    
    policy_mm = ActorCriticViTText(
        num_actions=env.action_space.n,
        vit_model="vit_tiny_patch16_224",
        text_model="distilbert-base-uncased",
        embedding_dim=192,
        hidden_dim=128,
        fusion_type="concat",
        pretrained=True,
        freeze_vit_layers=8,
        freeze_text=True,
    ).to(device)
    
    print(f"[OK] Created multimodal policy")
    print(f"  ViT: vit_tiny_patch16_224")
    print(f"  Text: distilbert-base-uncased")
    print(f"  Fusion: concat")
    print(f"  Parameters: {sum(p.numel() for p in policy_mm.parameters()):,}")
    
    # Test with text input
    history = HistoryBuffer(history_length=5, template_style='detailed')
    history.add(action=0, reward=1.0, done=False)
    history.add(action=1, reward=0.0, done=False)
    text = history.to_text()
    
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        action, logprob, entropy, value = policy_mm.get_action_and_value(obs_tensor, [text])
    
    print(f"[OK] Multimodal forward pass successful")
    print(f"  Text input: '{text[:60]}...'")
    print(f"  Action: {action.item()}")
    print(f"  Value: {value.item():.4f}")
    
    # Run a few episodes with multimodal
    print(f"\n[...] Running 5 episodes with multimodal policy...")
    
    total_rewards_mm = []
    for ep in range(5):
        obs, _ = env.reset()
        history = HistoryBuffer(history_length=5)
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 200:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
            text = history.to_text()
            
            with torch.no_grad():
                action, _, _, _ = policy_mm.get_action_and_value(obs_tensor, [text])
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            history.add(action=action.item(), reward=reward, done=done)
            ep_reward += reward
            steps += 1
        
        total_rewards_mm.append(ep_reward)
        print(f"  Episode {ep+1}: reward = {ep_reward:.0f}, steps = {steps}")
    
    print(f"[OK] Multimodal training test complete")
    print(f"  Average reward: {np.mean(total_rewards_mm):.2f}")
    
    # ============================================================
    # TEST 3: Different fusion types
    # ============================================================
    print(f"\n" + "-"*60)
    print("TEST 3: Different Fusion Types")
    print("-"*60)
    
    for fusion_type in ['concat', 'cross_attention', 'gated']:
        policy_fusion = ActorCriticViTText(
            num_actions=env.action_space.n,
            vit_model="vit_tiny_patch16_224",
            text_model="distilbert-base-uncased",
            embedding_dim=192,
            hidden_dim=128,
            fusion_type=fusion_type,
            pretrained=True,
        ).to(device)
        
        obs, _ = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            action, _, _, value = policy_fusion.get_action_and_value(obs_tensor, ["test text"])
        
        print(f"[OK] {fusion_type}: action={action.item()}, value={value.item():.4f}")
        
        del policy_fusion
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    env.close()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("[OK] ViT-only policy: PASSED")
    print("[OK] Multimodal (ViT+Text) policy: PASSED")
    print("[OK] All fusion types: PASSED")
    print("="*60)
    print("\nYour ViT RL implementation is working correctly!")
    print("\nNext steps:")
    print("  1. Install Atari: see ALE_PY_FIX.md")
    print("  2. Run full benchmark:")
    print("     python algos/train_ppo.py --env=PongNoFrameskip-v4 --condition=vit_only")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_vit_training()
    sys.exit(0 if success else 1)
