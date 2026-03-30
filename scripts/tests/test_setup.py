"""
Quick test script to verify installation and basic functionality
"""
import sys
import torch
import numpy as np


def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        import gymnasium
        print(f"✓ gymnasium {gymnasium.__version__}")
    except ImportError as e:
        print(f"✗ gymnasium: {e}")
        return False
    
    try:
        import timm
        print(f"✓ timm {timm.__version__}")
    except ImportError as e:
        print(f"✗ timm: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ opencv {cv2.__version__}")
    except ImportError as e:
        print(f"✗ opencv: {e}")
        return False
    
    print(f"✓ torch {torch.__version__}")
    print(f"✓ numpy {np.__version__}")
    
    return True


def test_environment():
    """Test environment creation"""
    print("\nTesting environment...")
    
    try:
        import gymnasium as gym
        # Test with CartPole (always available)
        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        print(f"✓ Basic environment (CartPole) works")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        env.close()
        
        # Try Atari if available
        try:
            from envs.atari_env import AtariEnvWrapper
            env = AtariEnvWrapper(env_name="PongNoFrameskip-v4", seed=42, vit_mode=True)
            obs, _ = env.reset()
            print(f"✓ Atari environment created successfully")
            print(f"  Observation shape: {obs.shape}")
            env.close()
        except Exception as e:
            print(f"⚠ Atari environments not available: {e}")
            print(f"  This is OK for testing, but needed for full benchmarks")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_vit_model():
    """Test ViT model creation"""
    print("\nTesting ViT model...")
    
    try:
        from models.vit_policy import ActorCriticViT
        
        model = ActorCriticViT(
            num_actions=6,
            vit_model="vit_base_patch16_224",
            embedding_dim=512,
            hidden_dim=256,
            pretrained=False,  # Don't download for quick test
            freeze_vit_layers=0,
        )
        
        # Test forward pass
        dummy_obs = torch.randn(2, 3, 224, 224)
        logits, value = model(dummy_obs)
        
        print(f"✓ ViT model created successfully")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Value shape: {value.shape}")
        return True
        
    except Exception as e:
        print(f"✗ ViT model test failed: {e}")
        return False


def test_text_components():
    """Test text encoding components"""
    print("\nTesting text components...")
    
    try:
        from utils.text_history import HistoryBuffer
        
        buffer = HistoryBuffer(history_length=5, template_style="detailed")
        
        # Add some transitions
        buffer.add(action=0, reward=1.0, done=False)
        buffer.add(action=1, reward=0.0, done=False)
        buffer.add(action=2, reward=-1.0, done=False)
        
        text = buffer.to_text()
        print(f"✓ Text history buffer working")
        print(f"  Sample text: {text[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ Text components test failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        return True
    else:
        print("⚠ CUDA not available - will use CPU (training will be slow)")
        return False


def main():
    print("="*80)
    print("BENCHMARK CODE TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("ViT Model", test_vit_model()))
    results.append(("Text Components", test_text_components()))
    results.append(("CUDA", test_cuda()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_critical_passed = all(passed for name, passed in results[:4])  # CUDA is optional
    
    if all_critical_passed:
        print("\n✓ All critical tests passed! Ready to run benchmarks.")
        return 0
    else:
        print("\n✗ Some critical tests failed. Please fix issues before running benchmarks.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
