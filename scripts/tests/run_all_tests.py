"""
Comprehensive Test Suite - Runs all tests and saves results
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import json


class TestRunner:
    def __init__(self, output_file="TEST_RESULTS.txt"):
        self.output_file = output_file
        self.results = []
        self.start_time = time.time()
        
    def log(self, message):
        """Log message to results"""
        print(message)
        self.results.append(message)
    
    def run_command(self, cmd, description, capture_output=True):
        """Run a command and capture results"""
        self.log("\n" + "="*80)
        self.log(f"TEST: {description}")
        self.log("="*80)
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("-"*80)
        
        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per test
                )
                self.log(result.stdout)
                if result.stderr:
                    self.log("STDERR:")
                    self.log(result.stderr)
                
                if result.returncode == 0:
                    self.log("\n✅ TEST PASSED")
                    return True
                else:
                    self.log(f"\n❌ TEST FAILED (Exit code: {result.returncode})")
                    return False
            else:
                result = subprocess.run(cmd)
                return result.returncode == 0
                
        except subprocess.TimeoutExpired:
            self.log("\n⏱️ TEST TIMEOUT (5 minutes exceeded)")
            return False
        except Exception as e:
            self.log(f"\n❌ TEST ERROR: {e}")
            return False
    
    def save_results(self):
        """Save all results to file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.results))
        
        # Also save JSON summary
        summary_file = self.output_file.replace('.txt', '_summary.json')
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': time.time() - self.start_time,
            'tests_run': len([r for r in self.results if 'TEST:' in r]),
            'tests_passed': len([r for r in self.results if '✅ TEST PASSED' in r]),
            'tests_failed': len([r for r in self.results if '❌ TEST FAILED' in r]),
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    print("="*80)
    print("COMPREHENSIVE BENCHMARK TEST SUITE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Create test runner
    runner = TestRunner("TEST_RESULTS.txt")
    
    runner.log("="*80)
    runner.log("COMPREHENSIVE BENCHMARK TEST SUITE")
    runner.log("="*80)
    runner.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    runner.log(f"Python: {sys.version}")
    runner.log("="*80)
    
    # Test 1: Setup verification
    passed_tests = 0
    total_tests = 0
    
    total_tests += 1
    if runner.run_command(
        [sys.executable, "test_setup.py"],
        "Setup Verification - Check all dependencies and imports"
    ):
        passed_tests += 1
    
    # Test 2: Quick training test (reduced steps)
    total_tests += 1
    runner.log("\n" + "="*80)
    runner.log("TEST: Quick Training Test (CartPole - 10k steps)")
    runner.log("="*80)
    runner.log("This tests the full training pipeline...")
    
    train_cmd = [
        sys.executable, "algos/train_ppo.py",
        "--env=CartPole-v1",
        "--condition=vit_only",
        "--seed=42",
        "--output-dir=results/test_suite/train",
        "--config=configs/quick_test.yaml",
    ]
    
    if runner.run_command(train_cmd, "Training Pipeline Test", capture_output=True):
        passed_tests += 1
    
    # Test 3: Model loading and evaluation
    checkpoint_path = Path("results/test_suite/train/final_model.pt")
    if checkpoint_path.exists():
        total_tests += 1
        eval_cmd = [
            sys.executable, "evaluation/eval.py",
            "--checkpoint=results/test_suite/train/final_model.pt",
            "--env=CartPole-v1",
            "--num-episodes=5",
            "--output=results/test_suite/eval_results.json",
        ]
        
        if runner.run_command(eval_cmd, "Policy Evaluation Test", capture_output=True):
            passed_tests += 1
    else:
        runner.log("\n⚠️ Skipping evaluation test - no checkpoint found")
    
    # Test 4: Test text history buffer
    total_tests += 1
    test_text_cmd = [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from utils.text_history import HistoryBuffer, ParallelHistoryBuffer

# Test single buffer
buffer = HistoryBuffer(history_length=5, template_style='detailed')
buffer.add(0, 1.0, False)
buffer.add(1, 0.5, False)
buffer.add(2, -0.5, False)
text = buffer.to_text()
assert len(text) > 0, "Text generation failed"
print(f"✓ Single buffer test passed. Sample: {text[:100]}")

# Test parallel buffer
pbuffer = ParallelHistoryBuffer(num_envs=4, history_length=3)
import numpy as np
pbuffer.add(
    actions=np.array([0, 1, 2, 1]),
    rewards=np.array([1.0, 0.0, -1.0, 0.5]),
    dones=np.array([False, False, False, False]),
    infos=[{}, {}, {}, {}]
)
texts = pbuffer.to_text_batch()
assert len(texts) == 4, "Parallel buffer failed"
print(f"✓ Parallel buffer test passed. {len(texts)} texts generated")

print("✓ All text history tests passed!")
"""]
    
    if runner.run_command(test_text_cmd, "Text History Buffer Test", capture_output=True):
        passed_tests += 1
    
    # Test 5: Test all fusion modules
    total_tests += 1
    test_fusion_cmd = [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
import torch
from models.fusion import ConcatFusion, CrossAttentionFusion, GatedFusion

batch_size = 4
vision_dim = 512
text_dim = 512

vision_feat = torch.randn(batch_size, vision_dim)
text_feat = torch.randn(batch_size, text_dim)

# Test concat fusion
concat = ConcatFusion(vision_dim, text_dim, output_dim=512)
out1 = concat(vision_feat, text_feat)
assert out1.shape == (batch_size, 512), f"Concat fusion failed: {out1.shape}"
print(f"✓ Concat fusion: {out1.shape}")

# Test cross-attention fusion
cross_attn = CrossAttentionFusion(vision_dim, text_dim, output_dim=512)
out2 = cross_attn(vision_feat, text_feat)
assert out2.shape == (batch_size, 512), f"Cross-attention fusion failed: {out2.shape}"
print(f"✓ Cross-attention fusion: {out2.shape}")

# Test gated fusion
gated = GatedFusion(vision_dim, text_dim, output_dim=512)
out3 = gated(vision_feat, text_feat)
assert out3.shape == (batch_size, 512), f"Gated fusion failed: {out3.shape}"
print(f"✓ Gated fusion: {out3.shape}")

print("✓ All fusion modules passed!")
"""]
    
    if runner.run_command(test_fusion_cmd, "Fusion Modules Test", capture_output=True):
        passed_tests += 1
    
    # Test 6: Test metrics collection
    total_tests += 1
    test_metrics_cmd = [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from evaluation.metrics import MetricsCollector, compute_sample_efficiency, compute_auc

# Test metrics collector
collector = MetricsCollector()
collector.add_training_metrics(100, {'episode_returns': 10.0, 'value_loss': 0.5})
collector.add_training_metrics(200, {'episode_returns': 15.0, 'value_loss': 0.3})
collector.add_eval_metrics(200, [12.0, 13.0, 14.0], [100, 110, 105])

summary = collector.get_summary()
print(f"✓ Metrics collector: {summary}")

# Test sample efficiency
returns = [5.0, 8.0, 12.0, 15.0, 18.0]
timesteps = [1000, 2000, 3000, 4000, 5000]
steps = compute_sample_efficiency(returns, timesteps, threshold=10.0)
assert steps == 3000, f"Sample efficiency failed: {steps}"
print(f"✓ Sample efficiency: {steps} steps to reach 10.0")

# Test AUC
auc = compute_auc(returns, timesteps)
print(f"✓ AUC: {auc:.2f}")

print("✓ All metrics tests passed!")
"""]
    
    if runner.run_command(test_metrics_cmd, "Metrics Collection Test", capture_output=True):
        passed_tests += 1
    
    # Test 7: Environment creation
    total_tests += 1
    test_env_cmd = [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
import gymnasium as gym
from envs.atari_env import make_parallel_envs

# Test CartPole
print("Testing CartPole environment...")
envs = make_parallel_envs('CartPole-v1', num_envs=4, seed=42, vit_mode=False)
obs, info = envs.reset()
print(f"✓ CartPole envs created: obs shape={obs.shape}")
actions = envs.action_space.sample()
obs, rewards, dones, truncated, info = envs.step(actions)
print(f"✓ CartPole step works: reward={rewards}")
envs.close()

# Test Atari if available
try:
    print("\\nTesting Atari environment...")
    from envs.atari_env import AtariEnvWrapper
    env = AtariEnvWrapper('PongNoFrameskip-v4', seed=42, vit_mode=True)
    obs, _ = env.reset()
    print(f"✓ Atari environment created: obs shape={obs.shape}")
    env.close()
except Exception as e:
    print(f"⚠ Atari not available (expected): {e}")

print("✓ Environment tests completed!")
"""]
    
    if runner.run_command(test_env_cmd, "Environment Creation Test", capture_output=True):
        passed_tests += 1
    
    # Final summary
    elapsed = time.time() - runner.start_time
    
    runner.log("\n" + "="*80)
    runner.log("TEST SUITE SUMMARY")
    runner.log("="*80)
    runner.log(f"Total tests run: {total_tests}")
    runner.log(f"Tests passed: {passed_tests}")
    runner.log(f"Tests failed: {total_tests - passed_tests}")
    runner.log(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    runner.log(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    runner.log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    runner.log("="*80)
    
    if passed_tests == total_tests:
        runner.log("\n🎉 ALL TESTS PASSED! Your benchmark code is fully functional!")
        exit_code = 0
    else:
        runner.log(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Check details above.")
        exit_code = 1
    
    # Save results
    summary = runner.save_results()
    
    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"Full results: {runner.output_file}")
    print(f"JSON summary: {runner.output_file.replace('.txt', '_summary.json')}")
    print("="*80)
    
    print("\nSUMMARY:")
    print(f"  Tests: {passed_tests}/{total_tests} passed")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Success rate: {(passed_tests/total_tests*100):.1f}%")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
