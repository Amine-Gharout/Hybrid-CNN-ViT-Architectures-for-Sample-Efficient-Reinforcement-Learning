"""
Generate plots for results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import argparse
 
# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300


def plot_learning_curves(
    results_dict: Dict[str, Dict],
    env_name: str,
    save_path: str = None,
):
    """Plot learning curves for all conditions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for condition, data in results_dict.items():
        timesteps = data['timesteps']
        returns = data['returns']
        
        ax.plot(timesteps, np.mean(returns, axis=0), label=condition, linewidth=2)
        ax.fill_between(
            timesteps,
            np.mean(returns, axis=0) - np.std(returns, axis=0),
            np.mean(returns, axis=0) + np.std(returns, axis=0),
            alpha=0.2
        )
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title(f'Learning Curves - {env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.close()


def plot_final_performance_bars(
    results_dict: Dict[str, List[float]],
    env_name: str,
    save_path: str = None,
):
    """Bar plot of final performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = list(results_dict.keys())
    means = [np.mean(results_dict[c]) for c in conditions]
    stds = [np.std(results_dict[c]) for c in conditions]
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='steelblue')
    
    # Color best performer
    best_idx = np.argmax(means)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Final Episode Return')
    ax.set_title(f'Final Performance - {env_name}')
    ax.grid(True, axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.close()


def plot_box_comparison(
    results_dict: Dict[str, List[float]],
    env_name: str,
    save_path: str = None,
):
    """Box plot comparison of all conditions"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = [results_dict[c] for c in results_dict.keys()]
    labels = list(results_dict.keys())
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Episode Return')
    ax.set_title(f'Performance Distribution - {env_name}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.close()


def load_results_from_dir(results_dir: Path) -> Dict[str, List[float]]:
    """Load evaluation results"""
    results = {}
    
    for json_file in results_dir.rglob("eval_results.json"):
        condition = json_file.parent.name
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'episode_returns' in data:
            if condition not in results:
                results[condition] = []
            results[condition].extend(data['episode_returns'])
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--env", type=str, default="Pong")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_dir}")
    results = load_results_from_dir(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} conditions")
    for condition, returns in results.items():
        print(f"  {condition}: {len(returns)} episodes, mean={np.mean(returns):.2f}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Bar plot
    plot_final_performance_bars(
        results,
        env_name=args.env,
        save_path=output_dir / f"{args.env}_performance_bars.png"
    )
    
    # Box plot
    plot_box_comparison(
        results,
        env_name=args.env,
        save_path=output_dir / f"{args.env}_boxplot.png"
    )
    
    print(f"\n✓ Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
