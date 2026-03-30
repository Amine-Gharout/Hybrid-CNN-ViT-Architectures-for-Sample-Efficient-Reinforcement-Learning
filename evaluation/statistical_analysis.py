"""
Statistical analysis of results
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple
import argparse
import json
from pathlib import Path


def compare_conditions(
    baseline_results: List[float],
    treatment_results: List[float],
    test: str = "mannwhitneyu",
    alpha: float = 0.05,
) -> Dict:
    """Statistical comparison between two conditions"""
    baseline = np.array(baseline_results)
    treatment = np.array(treatment_results)
    
    stats_dict = {
        'baseline_mean': np.mean(baseline),
        'baseline_std': np.std(baseline),
        'baseline_median': np.median(baseline),
        'treatment_mean': np.mean(treatment),
        'treatment_std': np.std(treatment),
        'treatment_median': np.median(treatment),
        'mean_diff': np.mean(treatment) - np.mean(baseline),
        'median_diff': np.median(treatment) - np.median(baseline),
    }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(baseline)**2 + np.std(treatment)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(treatment) - np.mean(baseline)) / pooled_std
    else:
        cohens_d = 0.0
    stats_dict['cohens_d'] = cohens_d
    
    # Statistical test
    if test == "mannwhitneyu":
        statistic, p_value = stats.mannwhitneyu(treatment, baseline, alternative='two-sided')
    elif test == "ttest":
        statistic, p_value = stats.ttest_ind(treatment, baseline)
    elif test == "wilcoxon" and len(baseline) == len(treatment):
        statistic, p_value = stats.wilcoxon(treatment, baseline)
    else:
        statistic, p_value = np.nan, np.nan
    
    stats_dict['test'] = test
    stats_dict['statistic'] = float(statistic) if not np.isnan(statistic) else None
    stats_dict['p_value'] = float(p_value) if not np.isnan(p_value) else None
    stats_dict['significant'] = p_value < alpha if not np.isnan(p_value) else None
    
    return stats_dict


def bootstrap_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval"""
    if statistic == "mean":
        point_est = np.mean(data)
        stat_func = np.mean
    elif statistic == "median":
        point_est = np.median(data)
        stat_func = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return point_est, lower, upper


def create_comparison_table(
    results_dict: Dict[str, List[float]],
    baseline_key: str = "vit_only",
) -> pd.DataFrame:
    """Create comparison table with all conditions"""
    rows = []
    
    baseline_results = results_dict.get(baseline_key, list(results_dict.values())[0])
    
    for condition, results in results_dict.items():
        mean, ci_low, ci_high = bootstrap_confidence_interval(
            np.array(results),
            statistic="mean"
        )
        
        row = {
            'Condition': condition,
            'Mean': f"{mean:.2f}",
            '95% CI': f"[{ci_low:.2f}, {ci_high:.2f}]",
            'Std': f"{np.std(results):.2f}",
            'Median': f"{np.median(results):.2f}",
        }
        
        if condition != baseline_key:
            comparison = compare_conditions(baseline_results, results)
            row['vs Baseline'] = f"{comparison['mean_diff']:+.2f}"
            row["Cohen's d"] = f"{comparison['cohens_d']:.3f}"
            row['p-value'] = f"{comparison['p_value']:.4f}" if comparison['p_value'] else "N/A"
            row['Significant'] = '✓' if comparison.get('significant') else '✗'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def load_results_from_dir(results_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load all evaluation results from directory"""
    results = {}
    
    for json_file in results_dir.rglob("*.json"):
        # Parse filename to extract condition and env
        parts = json_file.stem.split('_')
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract condition from path
        condition = None
        for part in json_file.parts:
            if 'vit' in part.lower():
                condition = part
                break
        
        if not condition:
            continue
        
        if condition not in results:
            results[condition] = {}
        
        # Store episode returns
        if 'episode_returns' in data:
            env_name = "unknown"
            for part in json_file.parts:
                if 'pong' in part.lower() or 'breakout' in part.lower():
                    env_name = part
                    break
            
            if env_name not in results[condition]:
                results[condition][env_name] = []
            
            results[condition][env_name].extend(data['episode_returns'])
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="vit_only")
    parser.add_argument("--output", type=str, default="results/statistical_analysis.csv")
    parser.add_argument("--test", type=str, default="mannwhitneyu", choices=["mannwhitneyu", "ttest", "wilcoxon"])
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print(f"Loading results from {results_dir}")
    
    # Load all results
    all_results = load_results_from_dir(results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"Found results for {len(all_results)} conditions")
    
    # Analyze each environment separately
    envs = set()
    for condition_results in all_results.values():
        envs.update(condition_results.keys())
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_tables = []
    
    for env in sorted(envs):
        print(f"\n{'='*80}")
        print(f"Environment: {env}")
        print(f"{'='*80}")
        
        env_results = {}
        for condition, condition_data in all_results.items():
            if env in condition_data:
                env_results[condition] = condition_data[env]
        
        if len(env_results) < 2:
            print(f"Skipping {env} - insufficient conditions")
            continue
        
        # Create comparison table
        table = create_comparison_table(env_results, baseline_key=args.baseline)
        table['Environment'] = env
        
        print(table.to_string(index=False))
        all_tables.append(table)
    
    # Save combined table
    if all_tables:
        combined_table = pd.concat(all_tables, ignore_index=True)
        combined_table.to_csv(output_path, index=False)
        print(f"\n✓ Statistical analysis saved to {output_path}")
    else:
        print("\nNo tables generated!")


if __name__ == "__main__":
    main()
