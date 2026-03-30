"""
Full benchmark execution script
"""
import argparse
import itertools
import subprocess
import yaml
from pathlib import Path
import sys
import time


BENCHMARK_ENVS = {
    "pong": "PongNoFrameskip-v4",
    "breakout": "BreakoutNoFrameskip-v4",
    "space_invaders": "SpaceInvadersNoFrameskip-v4",
    "ms_pacman": "MsPacmanNoFrameskip-v4",
}


def run_experiment(
    env_name: str,
    condition: str,
    seed: int,
    config_path: str,
    output_dir: str,
    use_wandb: bool = False,
    wandb_project: str = None,
):
    """Run a single experiment"""
    exp_name = f"{env_name.replace('NoFrameskip-v4', '')}_{condition}_seed{seed}"
    
    cmd = [
        sys.executable, "algos/train_ppo.py",
        f"--env={env_name}",
        f"--condition={condition}",
        f"--seed={seed}",
        f"--config={config_path}",
        f"--output-dir={output_dir}/{exp_name}",
    ]
    
    if use_wandb and wandb_project:
        cmd.extend([
            f"--wandb-project={wandb_project}",
            f"--wandb-name={exp_name}",
        ])
    
    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ {exp_name} completed successfully in {elapsed/3600:.2f} hours")
    else:
        print(f"\n✗ {exp_name} failed with code {result.returncode}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Config file path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--envs", nargs="+", default=None, help="Specific envs to run")
    parser.add_argument("--conditions", nargs="+", default=None, help="Specific conditions to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Specific seeds to run")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    parser.add_argument("--dry-run", action="store_true", help="Print experiments without running")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found")
        return 1
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get environments
    if args.envs:
        envs = {k: v for k, v in BENCHMARK_ENVS.items() if k in args.envs}
    else:
        envs = BENCHMARK_ENVS
    
    # Get conditions
    if args.conditions:
        conditions = {k: v for k, v in config['conditions'].items() if k in args.conditions}
    else:
        conditions = config['conditions']
    
    # Get seeds
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = config.get('seeds', [42])
    
    # Generate all combinations
    experiments = list(itertools.product(envs.keys(), conditions.keys(), seeds))
    
    print("\n" + "="*80)
    print("BENCHMARK CONFIGURATION")
    print("="*80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Environments: {list(envs.keys())}")
    print(f"Conditions: {list(conditions.keys())}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {args.output_dir}")
    print(f"WandB project: {args.wandb_project or 'None'}")
    print("="*80 + "\n")
    
    if args.dry_run:
        print("DRY RUN - Experiments that would be run:")
        for env_key, condition, seed in experiments:
            env_name = envs[env_key]
            exp_name = f"{env_key}_{condition}_seed{seed}"
            print(f"  - {exp_name}")
        return 0
    
    # Run experiments
    failed_experiments = []
    total_start = time.time()
    
    for i, (env_key, condition, seed) in enumerate(experiments, 1):
        env_name = envs[env_key]
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        try:
            returncode = run_experiment(
                env_name=env_name,
                condition=condition,
                seed=seed,
                config_path=str(config_path),
                output_dir=args.output_dir,
                use_wandb=args.wandb_project is not None,
                wandb_project=args.wandb_project,
            )
            
            if returncode != 0:
                failed_experiments.append(f"{env_key}_{condition}_seed{seed}")
                
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ ERROR in {env_key}/{condition}/seed{seed}: {e}")
            failed_experiments.append(f"{env_key}_{condition}_seed{seed}")
            continue
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"Completed: {len(experiments) - len(failed_experiments)}/{len(experiments)}")
    
    if failed_experiments:
        print(f"\nFailed experiments ({len(failed_experiments)}):")
        for exp in failed_experiments:
            print(f"  - {exp}")
        return 1
    else:
        print("\n✓ All experiments completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
