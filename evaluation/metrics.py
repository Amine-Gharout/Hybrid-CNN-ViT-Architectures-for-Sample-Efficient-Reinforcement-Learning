"""
Comprehensive metrics collection
"""
import numpy as np
from typing import Dict, List
import json


class MetricsCollector:
    """Collect and aggregate training/evaluation metrics"""
    
    def __init__(self):
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'value_loss': [],
            'policy_loss': [],
            'entropy': [],
            'learning_rate': [],
            'timesteps': [],
        }
        
        self.eval_metrics = {
            'eval_returns': [],
            'eval_lengths': [],
            'eval_timesteps': [],
        }
    
    def add_training_metrics(self, step: int, metrics: Dict):
        self.metrics['timesteps'].append(step)
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def add_eval_metrics(self, step: int, returns: List[float], lengths: List[int]):
        self.eval_metrics['eval_timesteps'].append(step)
        self.eval_metrics['eval_returns'].append(returns)
        self.eval_metrics['eval_lengths'].append(lengths)
    
    def get_summary(self) -> Dict:
        summary = {}
        
        if self.metrics['episode_returns']:
            returns = self.metrics['episode_returns']
            summary['final_train_return_mean'] = np.mean(returns[-100:])
            summary['final_train_return_std'] = np.std(returns[-100:])
        
        if self.eval_metrics['eval_returns']:
            all_eval_returns = [r for episode_returns in self.eval_metrics['eval_returns'] 
                               for r in episode_returns]
            summary['final_eval_return_mean'] = np.mean(all_eval_returns[-100:])
            summary['final_eval_return_std'] = np.std(all_eval_returns[-100:])
            summary['max_eval_return'] = np.max(all_eval_returns)
        
        return summary
    
    def save(self, path: str):
        data = {
            'metrics': self.metrics,
            'eval_metrics': self.eval_metrics,
            'summary': self.get_summary(),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.metrics = data['metrics']
        self.eval_metrics = data['eval_metrics']


def compute_sample_efficiency(
    returns: List[float],
    timesteps: List[int],
    threshold: float = 10.0,
) -> int:
    """Compute steps needed to reach performance threshold"""
    for i, (ret, step) in enumerate(zip(returns, timesteps)):
        if ret >= threshold:
            return step
    return -1


def compute_auc(
    returns: List[float],
    timesteps: List[int],
    normalize: bool = True,
) -> float:
    """Compute area under the learning curve"""
    if len(returns) < 2:
        return 0.0
    
    # Use trapezoid (new name) instead of deprecated trapz
    try:
        auc = np.trapezoid(returns, timesteps)
    except AttributeError:
        # Fallback for older numpy versions
        auc = np.trapz(returns, timesteps)
    
    if normalize:
        auc = auc / (timesteps[-1] - timesteps[0])
    
    return auc
