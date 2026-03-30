"""
Quick start training script - test with CartPole before Atari
"""
import subprocess
import sys

def main():
    print("="*80)
    print("QUICK START - Testing Benchmark Code with CartPole")
    print("="*80)
    print()
    
    # Test 1: ViT-only on CartPole
    print("Test 1: Training ViT-only policy on CartPole (5000 steps)...")
    print("-"*80)
    
    cmd1 = [
        sys.executable, "algos/train_ppo.py",
        "--env=CartPole-v1",
        "--condition=vit_only",
        "--seed=42",
        "--output-dir=results/quickstart/vit_only",
    ]
    
    print(f"Command: {' '.join(cmd1)}")
    print()
    
    try:
        subprocess.run(cmd1, check=True)
        print("\n✓ Test 1 passed!")
    except subprocess.CalledProcessError:
        print("\n✗ Test 1 failed!")
        return 1
    
    # Test 2: Multimodal on CartPole (if time permits)
    print("\n" + "="*80)
    print("Test 2: Training multimodal policy on CartPole (5000 steps)...")
    print("-"*80)
    
    cmd2 = [
        sys.executable, "algos/train_ppo.py",
        "--env=CartPole-v1",
        "--condition=vit_text_detailed",
        "--seed=42",
        "--output-dir=results/quickstart/vit_text",
    ]
    
    print(f"Command: {' '.join(cmd2)}")
    print()
    
    try:
        subprocess.run(cmd2, check=True)
        print("\n✓ Test 2 passed!")
    except subprocess.CalledProcessError:
        print("\n✗ Test 2 failed!")
        return 1
    
    # Evaluate
    print("\n" + "="*80)
    print("Test 3: Evaluating trained policies...")
    print("-"*80)
    
    cmd3 = [
        sys.executable, "evaluation/eval.py",
        "--checkpoint=results/quickstart/vit_only/final_model.pt",
        "--env=CartPole-v1",
        "--num-episodes=10",
        "--output=results/quickstart/eval_results.json",
    ]
    
    print(f"Command: {' '.join(cmd3)}")
    print()
    
    try:
        subprocess.run(cmd3, check=True)
        print("\n✓ Test 3 passed!")
    except subprocess.CalledProcessError:
        print("\n✗ Test 3 failed!")
        return 1
    
    # Success!
    print("\n" + "="*80)
    print("🎉 SUCCESS! All tests passed!")
    print("="*80)
    print()
    print("Your benchmark code is working correctly!")
    print()
    print("Next steps:")
    print("1. Install Atari ROMs (see WINDOWS_SETUP.md)")
    print("2. Run full benchmarks with: python scripts/run_full_benchmark.py")
    print("3. Analyze results with evaluation scripts")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
